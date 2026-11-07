<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Matrices as Linear Maps

*What does it mean to multiply a matrix by a vector, and why is that the central operation of machine learning?*

## The problem

A small neural network layer takes three numbers in and produces two numbers
out. Each output is a weighted sum of the inputs: the first output weights the
three inputs by $2, -1, 1$ and the second by $1, 3, 2$. That is two dot
products, each of the kind Chapter 3 defined, and they share the same input
vector. Writing them one after another is fine for two outputs. A real layer
has four thousand inputs and four thousand outputs: sixteen million weights,
four thousand dot products, repeated for every token of every sequence, for
every layer, for every training step. We need to treat the whole family of dot
products as **one object**, give it a name, and learn its rules, because three
questions cannot be answered about a pile of loose dot products.

1. **What is the object?** The weights are a table of numbers. But a table is
   only data; what makes it useful is what it *does* to a vector. We want the
   table and the action to be the same idea, so that facts about one are facts
   about the other.
2. **What happens when layers are stacked?** The output of one layer is the
   input of the next. If each layer is a family of dot products, what is the
   family of dot products that does both? Is there a rule for combining two
   tables into one, and is that rule a choice or is it forced?
3. **Can it be undone, and what does it lose?** Given the output of a layer,
   can we recover the input? Sometimes yes and sometimes never, and the
   difference decides whether information survives a layer, whether a system
   of equations has one solution, none, or infinitely many, and how many
   numbers a layer really needs.

A fourth question is practical and, for a machine that must pay for every
multiplication, decisive. **How much does it cost?** The answer turns out to
depend on the *order* in which we group a chain of products, by factors of
hundreds, with identical results. That is the first place the book's central
habit, estimating a computation before running it, pays for itself.

Chapter 3 gave us the vector, the dot product and the idea of a linear
combination. Chapter 2 gave us functions and their composition. This chapter
joins them: a **matrix** is a table of numbers that is also a function from
vectors to vectors, a **linear map**, and multiplying two matrices is composing
two such functions. We build everything from those two ideas and derive each
rule from them rather than defining it, so that the one rule that looks
arbitrary at first (how to multiply two matrices) turns out to be the only
rule that could work.

You may use from earlier chapters: vectors and the dot product (Chapter 3),
linear combinations (Chapter 3), functions, domains and composition
(Chapter 2), and sums (Chapter 1). Not assumed: determinants, eigenvalues,
anything about subspaces. The one idea we need from that territory, the
dimension of the set of outputs a map can reach, is built here from scratch
under the name **rank**. How a matrix is laid out in memory is Chapter 5, and
what a matrix hides inside it (rotations, low-rank structure) is Chapter 6.

## Before you begin

Five questions on ideas from earlier chapters, to answer from memory. The answers are at the end of the chapter. If one is hard, go back to the chapter named before reading on.

1. (Chapter 3, dot product.) Compute $(1, 2, 3) \cdot (4, 5, 6)$.
2. (Chapter 3, orthogonality.) What does it mean for the dot product of two nonzero vectors to be 0?
3. (Chapter 2, composition.) With $f(x) = x + 3$ and $g(x) = x^2$, evaluate $g(f(2))$. Which function is applied first?
4. (Chapter 2, functions.) Can a function give two different outputs for one input? Can two different inputs give the same output?
5. (Chapter 1, summation.) Evaluate $\sum_{l=1}^{3} l(l+1)$.

## A matrix is a table

A **matrix** with $m$ rows and $n$ columns is a rectangular table of real
numbers. We write it in bold capitals, $\mathbf{A}$, and its entry in row $i$,
column $j$ as $A_{ij}$:

$$
\mathbf{A} =
\begin{pmatrix}
A_{11} & A_{12} & \cdots & A_{1n} \\
A_{21} & A_{22} & \cdots & A_{2n} \\
\vdots & \vdots & & \vdots \\
A_{m1} & A_{m2} & \cdots & A_{mn}
\end{pmatrix}.
$$

The set of all such matrices is $\mathbb{R}^{m \times n}$, and the pair
$[m, n]$ is the matrix's shape, written with the rows first as in
[Notation](../notation.md). A matrix with $m = n$ is **square**. The shape is
part of the matrix: a $[2, 3]$ matrix and a $[3, 2]$ matrix are different
things even if they hold the same six numbers. As in Chapter 3, mathematics
counts from 1, so $A_{11}$ is the top left entry; array code counts from 0,
and where the text shows computed output (which uses 0-based labels like
"row 0") it says so.

There are two ways to read the same table, and the chapter's main work is
moving between them.

**By rows.** Row $i$ is a vector of $n$ numbers, $\mathbf{a}_{i}^{\top} =
(A_{i1}, \dots, A_{in})$. A matrix is a stack of $m$ row vectors. The layer
of the opening problem is $\mathbf{W}$ with rows $(2, -1, 1)$ and $(1, 3, 2)$:
each row is the weight vector of one output.

**By columns.** Column $j$ is a vector of $m$ numbers, $\mathbf{a}_{:j} =
(A_{1j}, \dots, A_{mj})$. A matrix is a row of $n$ column vectors laid side by
side. The same $\mathbf{W}$ has columns $(2, 1)$, $(-1, 3)$ and $(1, 2)$: each
column says what one input contributes to all the outputs.

A vector $\mathbf{x} \in \mathbb{R}^n$ is itself a matrix: a column of shape
$[n, 1]$ whenever we need it to be one. Chapter 3 wrote vectors as columns and
this chapter keeps to that, so that $\mathbf{x}^\top$ is a $[1, n]$ row.

## Linear maps

A matrix is a table, but we care about what it does. We want the most
general rule for turning a vector into another vector that behaves well with
the two operations of Chapter 3, addition and scaling. Here is the property.

A function $f: \mathbb{R}^n \to \mathbb{R}^m$ is a **linear map** if for all
vectors $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$ and every scalar $c$,

$$
f(\mathbf{x} + \mathbf{y}) = f(\mathbf{x}) + f(\mathbf{y}),
\qquad
f(c\,\mathbf{x}) = c\, f(\mathbf{x}).
$$

In words: adding then mapping equals mapping then adding, and scaling then
mapping equals mapping then scaling. A linear map does not care in which order
you apply it and the two basic vector operations. The two conditions combine
into one, $f(c\mathbf{x} + d\mathbf{y}) = c\,f(\mathbf{x}) + d\,f(\mathbf{y})$,
and by repeating it, a linear map commutes with every linear combination:

$$
f\Big(\sum_{j=1}^{k} c_j \mathbf{x}_j\Big) = \sum_{j=1}^{k} c_j f(\mathbf{x}_j).
$$

One consequence arrives at once. Taking $c = 0$ in the second condition gives
$f(\mathbf{0}) = \mathbf{0}$: **a linear map sends the zero vector to the zero
vector.** A function that does not do so is not linear, however straight its
graph looks.

The definition is only useful if it excludes things. Here are three functions
checked against it, one that passes and two that fail, on the layer's weights
$\mathbf{W}$ (rows $(2,-1,1)$ and $(1,3,2)$) and the vectors $\mathbf{x} =
(1, 2, 3)$ and $\mathbf{y} = (0, -1, 4)$.

<!-- worked: matrices.linearity_check -->
x = (1, 2, 3), y = (0, -1, 4), c = 3.

| expression | value |
|---|---:|
| A(x + y) | (8, 18) |
| Ax + Ay | (8, 18) |
| A(3x) | (9, 39) |
| 3(Ax) | (9, 39) |

Add a shift b = (10, -5), f(v) = Av + b:

| expression | value |
|---|---:|
| f(x + y) | (18, 13) |
| f(x) + f(y) | (28, 8) |
| f(0) | (10, -5) |

ReLU (negative entries become 0) on u = (1, -2), w = (-3, 1): relu(u + w) = (0, 0), relu(u) + relu(w) = (1, 1).
<!-- /worked -->

The first function, $\mathbf{x} \mapsto \mathbf{W}\mathbf{x}$ (the product is
defined in the next section, here read as the two dot products), satisfies
both conditions on the examples. Examples are not a proof, and the next
sections prove it. The second, $f(\mathbf{v}) = \mathbf{W}\mathbf{v} +
\mathbf{b}$ with a fixed shift $\mathbf{b}$, fails: $f(\mathbf{0}) =
\mathbf{b} \ne \mathbf{0}$, and $f(\mathbf{x} + \mathbf{y})$ differs from
$f(\mathbf{x}) + f(\mathbf{y})$ because the shift is added once on the left and
twice on the right. A linear map followed by a shift is an **affine map**.
Affine maps are not linear, but they are linear *plus one constant*, and the
trick at the end of this chapter removes even that difference. The third,
ReLU from Chapter 2 applied to each entry, fails because bending at zero does
not commute with adding: $\operatorname{relu}(u + w)$ is not
$\operatorname{relu}(u) + \operatorname{relu}(w)$ when the signs differ.

That last failure is no accident of the example. It is why a network is
built from linear maps *and* bends: a pile of linear maps is secretly just one
linear map (we will see exactly why in a few sections), so all the expressive
power of a network beyond a single linear map comes from the functions between
the layers. The matrix is the part of a network that is easy to analyse, and
this chapter is about that part.

## The matrix-vector product, derived

Which linear maps are there? One question settles all of them. Take any linear
map $f: \mathbb{R}^n \to \mathbb{R}^m$ and the **unit vectors**
$\mathbf{e}_1, \dots, \mathbf{e}_n$ of $\mathbb{R}^n$, where $\mathbf{e}_j$ has
a 1 in position $j$ and 0 elsewhere. Every vector is a linear combination of
them, because writing out its entries is writing out the weights:

$$
\mathbf{x} = x_1 \mathbf{e}_1 + x_2 \mathbf{e}_2 + \dots + x_n \mathbf{e}_n .
$$

Apply $f$ and use linearity, which lets $f$ pass through the sum and the
weights:

$$
f(\mathbf{x}) = x_1 f(\mathbf{e}_1) + x_2 f(\mathbf{e}_2) + \dots + x_n f(\mathbf{e}_n).
$$

This is the key fact of the chapter. **A linear map is completely determined by
what it does to the $n$ unit vectors.** If we know the $n$ vectors
$f(\mathbf{e}_1), \dots, f(\mathbf{e}_n)$, each a vector of $m$ numbers, we know
$f$ everywhere, because every output is a linear combination of them with the
input's entries as weights. Store those $n$ vectors as the columns of a table.
That table has $m$ rows and $n$ columns, and it is a matrix: the matrix of $f$.
Conversely, given any $[m, n]$ table $\mathbf{A}$, the rule "the output is
the combination of $\mathbf{A}$'s columns, weighted by the input" is a linear
map, because a combination of columns whose weights are the entries of
$\mathbf{x} + \mathbf{y}$ is the sum of the two combinations, and likewise for
scaling. So **matrices and linear maps $\mathbb{R}^n \to \mathbb{R}^m$ are the
same thing**, and the table is a complete description of the map.

That gives the definition. The **matrix-vector product** of
$\mathbf{A} \in \mathbb{R}^{m \times n}$ and $\mathbf{x} \in \mathbb{R}^n$ is the
vector $\mathbf{A}\mathbf{x} \in \mathbb{R}^m$ with

$$
\mathbf{A}\mathbf{x} \;=\; x_1\, \mathbf{a}_{:1} + x_2\, \mathbf{a}_{:2} + \dots + x_n\, \mathbf{a}_{:n},
\qquad\text{so}\qquad
(\mathbf{A}\mathbf{x})_i = \sum_{j=1}^{n} A_{ij}\, x_j .
$$

The second form is the first read one output entry at a time: entry $i$ of the
combination collects the $i$-th entry of every column, weighted by the $x_j$.
Notice what it is. $\sum_j A_{ij} x_j$ is the dot product of row $i$ of
$\mathbf{A}$ with $\mathbf{x}$. So the same product has two descriptions, and
both are worth knowing.

**The row view: a list of dot products.** Entry $i$ of $\mathbf{A}\mathbf{x}$
is $\mathbf{a}_i \cdot \mathbf{x}$, the dot product of row $i$ with the input.
This is the view of the opening problem: each row is a weight vector (a
neuron, in the language of Chapter 3) and the product evaluates all $m$ of them
on the same input. It is the view to use when asking *how does one output
depend on the input* (look at one row) or how to compute the product (it is $m$
independent dot products, which a machine can run side by side).

**The column view: a weighted sum of columns.** The whole output is
$\sum_j x_j \mathbf{a}_{:j}$: each input entry scales its column and the scaled
columns are added. It is the view to use when asking *what can this map
produce*: every output is a combination of the columns, so the outputs are
confined to what those $n$ columns can combine to, a fact the section on rank
turns into a number. It is also how to read a column: $\mathbf{a}_{:j}$ is the
output when the input is $\mathbf{e}_j$, "what input $j$ alone does". A
matrix with $0$s except for one nonzero column keeps one input and ignores the
rest.

The numbers on the opening layer show both views giving one answer, entry by
entry in rows and as weighted columns:

<!-- worked: matrices.matvec_two_views -->
A = [[2, -1, 1], [1, 3, 2]], shape [2, 3]; x = (1, 2, 3).

Row view: entry i of Ax is row i of A dotted with x.

| entry | row of A | dot with x | value |
|---|---|---|---:|
| row 0 | (2, -1, 1) | 2*1 + -1*2 + 1*3 | 3 |
| row 1 | (1, 3, 2) | 1*1 + 3*2 + 2*3 | 13 |

Column view: Ax is the sum of the columns of A, column j weighted by x[j].

| term | column of A | weight x[j] | weighted column |
|---|---|---|---:|
| column 0 | (2, 1) | 1 | (2, 1) |
| column 1 | (-1, 3) | 2 | (-2, 6) |
| column 2 | (1, 2) | 3 | (3, 6) |

Sum of the weighted columns: (3, 13). Row view: (3, 13).
<!-- /worked -->

The first view cost two dot products of three multiplications each; the second
scaled three columns of two entries and added them. Both are six
multiply-adds. They agree because they are one formula, summed in two different
orders.

The definition obeys a shape rule that every later section leans on. The
product $\mathbf{A}\mathbf{x}$ exists only when the number of columns of
$\mathbf{A}$ equals the dimension of $\mathbf{x}$: shape $[m, n]$ times shape
$[n]$ gives shape $[m]$. The inner $n$ must match and disappears, the outer $m$
survives. Mismatched shapes are not a rounding problem or a missing zero; they
mean the program is wrong, and the pseudocode below refuses them.

Last, the definition proves that the product is linear. If $\mathbf{x}$ and
$\mathbf{y}$ are inputs, then $(\mathbf{A}(\mathbf{x} + \mathbf{y}))_i =
\sum_j A_{ij}(x_j + y_j) = \sum_j A_{ij} x_j + \sum_j A_{ij} y_j$, which is
entry $i$ of $\mathbf{A}\mathbf{x} + \mathbf{A}\mathbf{y}$, and the same
step with $c x_j$ gives scaling. The checks in the previous section were
therefore not luck.

## The picture: what a matrix does to a square

In two dimensions a linear map can be drawn. Take the unit square with corners
$(0,0)$, $(1,0)$, $(1,1)$, $(0,1)$, spanned by $\mathbf{e}_1$ and
$\mathbf{e}_2$, and a grid of lines drawn on it. Apply the map to every point.
By the column view the image of the point $(s, t)$ is $s\,\mathbf{a}_{:1} +
t\,\mathbf{a}_{:2}$: the image of $\mathbf{e}_1$ and the image of $\mathbf{e}_2$
are the two columns, and everything else is built from them. So the square
becomes the **parallelogram** spanned by the two columns, and the grid of
straight, evenly spaced lines becomes a grid of straight, evenly spaced
lines. That straightness is linearity made visible: a map that bent a line
would break $f(\mathbf{x} + \mathbf{y}) = f(\mathbf{x}) + f(\mathbf{y})$.
The origin stays where it is, as it must.

<!-- worked: matrices.columns_are_images -->
A = [[2, 1], [1/2, 3/2]]. A e1 = (2, 1/2) and A e2 = (1, 3/2) are its columns.

| corner x | x[0] A e1 + x[1] A e2 | A x |
|---|---|---|
| (0, 0) | 0*(2, 1/2) + 0*(1, 3/2) | (0, 0) |
| (1, 0) | 1*(2, 1/2) + 0*(1, 3/2) | (2, 1/2) |
| (1, 1) | 1*(2, 1/2) + 1*(1, 3/2) | (3, 2) |
| (0, 1) | 0*(2, 1/2) + 1*(1, 3/2) | (1, 3/2) |

The image of the unit square is a parallelogram of area 5/2 (shoelace formula over the four images); the unit square has area 1.
<!-- /worked -->

![Two maps applied to the unit square and a grid on it. Left: the matrix with columns (2, 0.5) and (1, 1.5) turns the square into a tilted parallelogram of area 5/2, with the images of the unit vectors drawn as arrows. Right: a matrix with columns (1, 0.5) and (2, 1), whose second column is twice the first, sends the whole square onto a single line through the origin, area 0.](img/linear-map-square.svg)

**Figure 1.** Left: the map with columns $(2, \tfrac12)$ and $(1, \tfrac32)$.
Both images of the unit vectors are drawn, the square becomes the shaded
parallelogram and each grid line becomes a parallel line. Right: a map whose
second column is twice the first. The images of $\mathbf{e}_1$ and
$\mathbf{e}_2$ lie on one line, so every combination of them does too, and the
square is flattened to a segment. All points of the figure are computed with
the matrix-vector product of this chapter.

The right panel is the other half of the story. When the columns point along
one line, the map squashes the plane onto that line: two different inputs can
land on the same output, and the information that told them apart is gone. How
often that happens, and what it costs, is the subject of the second half of
the chapter. The area of the parallelogram in the left panel, computed by the
shoelace formula over its four corners, is $\tfrac52$, and the right panel
has area $0$; keep both numbers, the next sections explain them.

## Composition forces matrix multiplication

Two layers in a row compute $\mathbf{B}(\mathbf{A}\mathbf{x})$: first map by
$\mathbf{A}$, then map the result by $\mathbf{B}$. By Chapter 2 that is the
composition $g \circ f$ of $f(\mathbf{x}) = \mathbf{A}\mathbf{x}$ and
$g(\mathbf{y}) = \mathbf{B}\mathbf{y}$. The shapes must chain: $f$ goes from
$\mathbb{R}^n$ to $\mathbb{R}^k$ so $\mathbf{A}$ is $[k, n]$, and $g$ goes
from $\mathbb{R}^k$ to $\mathbb{R}^m$, so $\mathbf{B}$ is $[m, k]$.

The composition of two linear maps is linear. Adding inputs, $g(f(\mathbf{x} +
\mathbf{y})) = g(f(\mathbf{x}) + f(\mathbf{y})) = g(f(\mathbf{x})) +
g(f(\mathbf{y}))$, using linearity of $f$ and then of $g$; scaling is the same.
Therefore $g \circ f$ is a linear map $\mathbb{R}^n \to \mathbb{R}^m$ and, by
the previous section, **it has a matrix**. Which one? The section tells us how
to find it without guessing: its columns are the images of the unit vectors.
Column $j$ is $g(f(\mathbf{e}_j))$, which is $\mathbf{B}$ applied to
$\mathbf{A}\mathbf{e}_j$, which is column $j$ of $\mathbf{A}$. So

$$
\text{column } j \text{ of the composite} = \mathbf{B}\,\mathbf{a}_{:j}.
$$

The matrix of the composition is found by multiplying $\mathbf{B}$ into each
column of $\mathbf{A}$. Take that as the definition of the **matrix product**
$\mathbf{B}\mathbf{A}$, of shape $[m, n]$, and expand the matrix-vector product
inside it to see its entries:

$$
(\mathbf{B}\mathbf{A})_{ij} \;=\; (\mathbf{B}\,\mathbf{a}_{:j})_i
\;=\; \sum_{l=1}^{k} B_{il}\, A_{lj}.
$$

Entry $(i, j)$ of the product is the dot product of row $i$ of $\mathbf{B}$
with column $j$ of $\mathbf{A}$. This is the textbook formula for
**matrix multiplication**, and the point is that it is not a convention.
Nothing else would work: we asked for the matrix whose action on every
$\mathbf{x}$ is "apply $\mathbf{A}$, then apply $\mathbf{B}$", and the unit
vectors forced each column. The formula looks odd (why rows of the left matrix
against columns of the right one?) until it is seen as the composition rule
written in entries.

Here it is on two real maps: $f$ from $\mathbb{R}^2$ to $\mathbb{R}^3$ with a
$[3, 2]$ matrix, and $g$ from $\mathbb{R}^3$ back to $\mathbb{R}^2$ with a
$[2, 3]$ matrix. The table feeds the unit vectors through both stages and reads
off the columns.

<!-- worked: matrices.composition_is_product -->
f(x) = Ax with A = [[1, 2], [0, 1], [3, -1]] (shape [3, 2]); g(y) = By with B = [[2, 0, 1], [1, 1, -1]] (shape [2, 3]).

Feed the unit vectors through f, then g. The results are the columns of the matrix of g after f.

| input | f(input) = A input | g(f(input)) |
|---|---|---|
| e1 = (1, 0) | (1, 0, 3) | (5, -2) |
| e2 = (0, 1) | (2, 1, -1) | (3, 4) |

So that matrix has columns (5, -2) and (3, 4): C = [[5, 3], [-2, 4]], shape [2, 2], and C[i][j] = sum over l of B[i][l] A[l][j].

Check on x = (2, -1): g(f(x)) = (7, -8), Cx = (7, -8).

Entry C[0][1] = 2*2 + 0*1 + 1*-1 = 3.
<!-- /worked -->

The matrix $\mathbf{C} = \mathbf{B}\mathbf{A}$ built column by column, and the
two-stage evaluation, agree on every input we try: the product is by
construction the composition. The shape rule is the chain rule of shapes:
$[m, k] \times [k, n] = [m, n]$. The shared $k$ is the dimension of the
intermediate space, the "hidden" layer in the middle, and it disappears from
the product's shape. The product is not defined when the inner sizes disagree,
for the same reason a composition is not defined when the output set of the
first map is not the input set of the second.

![A dataflow diagram: matrix A of shape m by k and matrix B of shape k by n both feed a node computing each entry of C as row i of A dotted with column j of B, which takes k multiply-adds, producing C of shape m by n; the shared size k is labelled as disappearing.](img/matmul-shapes.svg)

**Figure 2.** The shapes in $[m, k] \times [k, n] = [m, n]$. The inner size $k$ is
shared and vanishes; the outer sizes survive. Each of the $m n$ output
entries is one dot product of length $k$.

## What a product costs

The picture of Figure 2 is also a cost estimate. Each of the $m \cdot n$ entries
of the product is a dot product of length $k$, which is $k$ multiplications and
$k - 1$ additions (Chapter 3), about $k$ **multiply-adds** (one multiplication
fused with one addition, the unit of count used from now on). So multiplying a
$[m, k]$ matrix by a $[k, n]$ matrix costs

$$
m \cdot n \cdot k \ \text{multiply-adds},
\qquad\text{or } 2mnk \text{ floating-point operations}
$$

by the convention of [Notation](../notation.md). The matrix-vector product is
the special case $n = 1$ and costs $mk$.

<!-- worked: matrices.shapes_and_cost -->
| product | result shape | outputs m*n | terms per output k | multiply-adds m*n*k | FLOP 2*m*n*k |
|---|---:|---:|---:|---:|---:|
| [2, 3] x [3, 4] | [2, 4] | 8 | 3 | 24 | 48 |
| [1, 512] x [512, 512] | [1, 512] | 512 | 512 | 262,144 | 524,288 |
| [512, 1] x [1, 512] | [512, 512] | 262,144 | 1 | 262,144 | 524,288 |
| [512, 512] x [512, 512] | [512, 512] | 262,144 | 512 | 134,217,728 | 268,435,456 |
| [128, 4096] x [4096, 4096] | [128, 4096] | 524,288 | 4,096 | 2,147,483,648 | 4,294,967,296 |

Counting the inner-loop iterations of a plain triple loop on a [2, 3] x [3, 4] example gives 24 multiply-adds, the value of m*n*k = 2*3*4.
<!-- /worked -->

Three things to notice. The count depends on all three sizes and is the
*product* of them, so doubling every dimension multiplies the cost by eight.
The matrix-vector case (second row) is far cheaper than the square product
(fourth row) at the same $k$, by a factor of $n$: this will matter in a moment.
And the count makes no mention of the numbers, only of the shapes, so it can be
computed before a single weight exists. That is the practice Chapter 16 builds
into a method; this chapter uses it once, in the next section, to settle an
argument about order.

## Order: commutativity, associativity, and the cost of grouping

Matrix multiplication has two properties that look like those of numbers and one
that does not.

**It is not commutative.** In general $\mathbf{A}\mathbf{B} \ne
\mathbf{B}\mathbf{A}$. For some shapes the question does not even arise, since
the two products need not both be defined. When they are, the answers differ:
composing "swap the two coordinates, then apply $\mathbf{A}$" is not
composing "apply $\mathbf{A}$, then swap":

<!-- worked: matrices.noncommutative -->
A = [[1, 2], [3, 4]], B = [[0, 1], [1, 0]] (B swaps the two coordinates).

| product | value | what happened |
|---|---|---|
| A B | [[2, 1], [4, 3]] | columns of A swapped |
| B A | [[3, 4], [1, 2]] | rows of A swapped |

With P = [[2, 0, 1], [1, 1, -1]] of shape [2, 3] and Q = [[1, 2], [0, 1], [3, -1]] of shape [3, 2]: P Q has shape [2, 2] and Q P has shape [3, 3].
<!-- /worked -->

Multiplying on the right by the swap permutes the *columns* of $\mathbf{A}$;
multiplying on the left permutes its *rows*. That is not two names for one
thing: the column view of $\mathbf{A}\mathbf{B}$ (every column of the product
is $\mathbf{A}$ times a column of $\mathbf{B}$) and the row view of
$\mathbf{B}\mathbf{A}$ (every row is a row of $\mathbf{B}$ times $\mathbf{A}$)
describe two different operations. Functions do not commute either, which is
all this says: in Chapter 2, $f \circ g$ and $g \circ f$ were different
functions, and the matrix product is the composition.

**It is associative.** $(\mathbf{A}\mathbf{B})\mathbf{C} =
\mathbf{A}(\mathbf{B}\mathbf{C})$ whenever the shapes chain. The proof is one
line from composition: both sides are the matrix of the map $\mathbf{x}
\mapsto \mathbf{A}(\mathbf{B}(\mathbf{C}\mathbf{x}))$, apply $\mathbf{C}$ then
$\mathbf{B}$ then $\mathbf{A}$, and two matrices that act the same way on every
unit vector have the same columns, hence are equal. Product is also
distributive, $\mathbf{A}(\mathbf{B} + \mathbf{C}) = \mathbf{A}\mathbf{B} +
\mathbf{A}\mathbf{C}$, by linearity of $\mathbf{A}$ applied column by column.
Because grouping does not change the result we may write $\mathbf{A}\mathbf{B}
\mathbf{C}$ without brackets, and that freedom is exactly what makes the next
point valuable.

**Grouping changes the cost, not the answer.** Two groupings of the same product
are the same matrix, but they are not the same amount of work. Take two
$[512, 512]$ matrices $\mathbf{A}$ and $\mathbf{B}$ and a vector $\mathbf{x}$ of
512 entries. The expression $\mathbf{A}\mathbf{B}\mathbf{x}$ may be computed as
$(\mathbf{A}\mathbf{B})\mathbf{x}$: multiply the two matrices, then the result
by the vector. Or as $\mathbf{A}(\mathbf{B}\mathbf{x})$: two matrix-vector
products. Chain of three matrices of different shapes shows the effect too.

<!-- worked: matrices.association_costs -->
A and B are [512, 512], x is [512, 1]. Cost in multiply-adds:

| order | first product | second product | total |
|---|---|---|---|
| (A B) x | A B: [512,512]x[512,512] = 134,217,728 | (AB) x: [512,512]x[512,1] = 262,144 | 134,479,872 |
| A (B x) | B x: [512,512]x[512,1] = 262,144 | A (Bx): [512,512]x[512,1] = 262,144 | 524,288 |

The ratio of the two totals is 256.5.

A is [10, 100], B is [100, 5], C is [5, 50]:

| order | first product | second product | total |
|---|---|---|---|
| (A B) C | A B -> [10, 5]: 5,000 | (AB) C -> [10, 50]: 2,500 | 7,500 |
| A (B C) | B C -> [100, 50]: 25,000 | A (BC) -> [10, 50]: 50,000 | 75,000 |

Integer matrices (entry (i, j), counting from 0, is (3i + 2j) mod 7 - 3 for P, (i + 4j) mod 5 - 2 for Q, (2i + j) mod 6 - 2 for R):

P = [[-3, -1, 1, 3], [0, 2, -3, -1], [3, -2, 0, 2]] [3, 4]

Q = [[-2, 2], [-1, -2], [0, -1], [1, 0]] [4, 2]

R = [[-2, -1, 0, 1, 2], [0, 1, 2, 3, -2]] [2, 5]

Both groupings, (P Q) R and P (Q R), give the same [3, 5] result [[-20, -15, -10, -5, 30], [6, 2, -2, -6, -4], [4, 12, 20, 28, -24]], as associativity promises.
<!-- /worked -->

The first order is hundreds of times more expensive than the second for the
same answer: it multiplies two big matrices to throw away almost everything
the product contains, then applies the result to a single vector. The cost rule
$m n k$ explains why. Multiplying the matrix by the vector first makes every
intermediate result a vector, and a matrix-vector product is $n = 1$, with cost
only $mk$. In the chain of three, the best order is the one that makes the
*intermediate* shapes small: $(\mathbf{A}\mathbf{B})\mathbf{C}$ produces a
$[10, 5]$ intermediate and costs $7500$, while $\mathbf{A}(\mathbf{B}\mathbf{C})$
produces a $[100, 50]$ intermediate and costs ten times that. The rule of
thumb is to group so that the intermediate results are as small as possible,
and for a long chain the best grouping can be found by a small search
(Exercise 5). This is not an optimisation to apply late: a model that applies
the same low-rank change to a thousand vectors, or an attention step that
contracts in the wrong order, has paid this factor, and no profiler will tell
you the source because every individual operation looks reasonable.

## The transpose

A **transpose** exchanges rows and columns. The transpose of $\mathbf{A}$ of
shape $[m, n]$ is $\mathbf{A}^\top$ of shape $[n, m]$ with

$$
(\mathbf{A}^\top)_{ij} = A_{ji}.
$$

Row $i$ of $\mathbf{A}^\top$ is column $i$ of $\mathbf{A}$. Transposing twice
returns the original, $(\mathbf{A}^\top)^\top = \mathbf{A}$. For a vector,
transposing turns a column $[n, 1]$ into a row $[1, n]$, and with it the dot
product of Chapter 3 becomes a matrix product, $\mathbf{x}\cdot\mathbf{y} =
\mathbf{x}^\top \mathbf{y}$ of shape $[1, n]\times[n, 1] = [1, 1]$.

How does a transpose interact with a product? Look at one entry of
$(\mathbf{A}\mathbf{B})^\top$. By the definitions,

$$
\big((\mathbf{A}\mathbf{B})^\top\big)_{ij} = (\mathbf{A}\mathbf{B})_{ji}
= \sum_{l} A_{jl} B_{li}
= \sum_{l} (\mathbf{B}^\top)_{il} (\mathbf{A}^\top)_{lj}
= (\mathbf{B}^\top \mathbf{A}^\top)_{ij}.
$$

The middle step only reorders each product of two numbers, and then reads the
two factors as entries of the transposes. Hence

$$
(\mathbf{A}\mathbf{B})^\top = \mathbf{B}^\top \mathbf{A}^\top .
$$

The order **reverses**. The shapes force it: if $\mathbf{A}$ is $[m, k]$ and
$\mathbf{B}$ is $[k, n]$, then $\mathbf{A}^\top\mathbf{B}^\top$ is
$[k, m]\times[n, k]$, usually not even defined, whereas $\mathbf{B}^\top
\mathbf{A}^\top$ is $[n, k]\times[k, m] = [n, m]$, the shape of
$(\mathbf{A}\mathbf{B})^\top$. The same reversal appears again for inverses
(next section) and, in Volume III, in the backward pass.

<!-- worked: matrices.transpose_rules -->
A = [[1, 2], [0, 1], [3, -1]] [3, 2], B = [[2, 0, 1], [1, 1, -1]] [2, 3].

| expression | shape | value |
|---|---|---:|
| A B | [3, 3] | [[4, 2, -1], [1, 1, -1], [5, -1, 4]] |
| (A B)^T | [3, 3] | [[4, 1, 5], [2, 1, -1], [-1, -1, 4]] |
| B^T A^T | [3, 3] | [[4, 1, 5], [2, 1, -1], [-1, -1, 4]] |
| A^T B^T | [2, 2] | [[5, -2], [3, 4]] |

Adjoint rule for M = B, x = (1, -1, 2), y = (3, 0): (Mx).y = (4, -2).(3, 0) = 12; x.(M^T y) = (1, -1, 2).(6, 0, 3) = 12.
<!-- /worked -->

The table checks the rule on a $[3, 2]$ times $[2, 3]$ product, and shows the
wrong order giving a matrix of the wrong shape altogether. The last line shows
the transpose's defining role: for any matrix $\mathbf{M}$,

$$
(\mathbf{M}\mathbf{x}) \cdot \mathbf{y} \;=\; \mathbf{x} \cdot (\mathbf{M}^\top \mathbf{y}),
$$

which follows from the definitions by writing both sides as $\sum_i \sum_j
y_i M_{ij} x_j$. The transpose is the matrix that lets you move a map from one
side of a dot product to the other. It is why the backward pass of a layer
multiplies by the weights' transpose: asking how a change in the output moves
the input uses the same map read in the other direction.

## Identity and inverse

Is there a matrix that does nothing? The **identity matrix** $\mathbf{I}_n$ of
shape $[n, n]$ has ones on the diagonal and zeros elsewhere. Its columns are the
unit vectors, so by the column view $\mathbf{I}\mathbf{x} = x_1\mathbf{e}_1 +
\dots + x_n\mathbf{e}_n = \mathbf{x}$: it is the map that leaves every vector
alone. It is the unit of the product: $\mathbf{I}\mathbf{A} = \mathbf{A} =
\mathbf{A}\mathbf{I}$, with $\mathbf{I}$ of the right size on each side.

If a map does something, can we undo it? A square matrix $\mathbf{A}$ has an
**inverse** $\mathbf{A}^{-1}$ if

$$
\mathbf{A}^{-1}\mathbf{A} = \mathbf{I} = \mathbf{A}\mathbf{A}^{-1},
$$

so that the map $\mathbf{A}^{-1}$ undoes the map $\mathbf{A}$ and vice
versa (Chapter 2: an inverse function). Solving $\mathbf{A}\mathbf{x} =
\mathbf{y}$ for $\mathbf{x}$ is then multiplying both sides by
$\mathbf{A}^{-1}$: $\mathbf{x} = \mathbf{A}^{-1}\mathbf{y}$. A matrix with an
inverse is **invertible**; one without is **singular**. Only square matrices
can be inverted here: the outputs of a $[3, 2]$ map fill at most a plane inside
three-dimensional space, so most points of space are not outputs of anything,
and a $[2, 3]$ map must merge points to fit three dimensions into two, so it
cannot be undone.

**The inverse of a product** follows from the same logic as the transpose. To
undo "first $\mathbf{B}$, then $\mathbf{A}$" you must undo $\mathbf{A}$ first,
then $\mathbf{B}$: $(\mathbf{A}\mathbf{B})^{-1} = \mathbf{B}^{-1}
\mathbf{A}^{-1}$. Check: $(\mathbf{A}\mathbf{B})(\mathbf{B}^{-1}\mathbf{A}^{-1})
= \mathbf{A}(\mathbf{B}\mathbf{B}^{-1})\mathbf{A}^{-1} = \mathbf{A}
\mathbf{I}\mathbf{A}^{-1} = \mathbf{I}$, using associativity. Socks go on
before shoes and come off in the reverse order.

### The $2 \times 2$ case, derived

For a $[2, 2]$ matrix the inverse can be found by solving the system
directly. Write

$$
\mathbf{A} = \begin{pmatrix} a & b \\ c & d \end{pmatrix},
\qquad \mathbf{A}\mathbf{x} = \mathbf{y}:
\qquad
\begin{aligned} a x_1 + b x_2 &= y_1, \\ c x_1 + d x_2 &= y_2. \end{aligned}
$$

To remove $x_2$, multiply the first equation by $d$ and the second by $b$ and
subtract: $(ad - bc)\, x_1 = d\, y_1 - b\, y_2$. To remove $x_1$, multiply the
first by $c$ and the second by $a$ and subtract the other way:
$(ad - bc)\, x_2 = a\, y_2 - c\, y_1$. Both are divided by the same number
$\Delta = ad - bc$. If $\Delta \ne 0$ we divide and read off $\mathbf{x} =
\mathbf{A}^{-1}\mathbf{y}$:

$$
\mathbf{A}^{-1} = \frac{1}{ad - bc}\begin{pmatrix} d & -b \\ -c & a \end{pmatrix}.
$$

Multiplying out $\mathbf{A}\mathbf{A}^{-1}$ gives $\mathbf{I}$, as the check
below confirms. The number $\Delta$ decides everything: if $\Delta \ne 0$
there is exactly one solution for every $\mathbf{y}$, and if $\Delta = 0$ the
two equations either contradict each other or say the same thing twice, so for
some $\mathbf{y}$ there is no solution and for others infinitely many. It is the
number that was hiding in Figure 1: the left map has $\Delta = 2 \cdot
\tfrac32 - 1 \cdot \tfrac12 = \tfrac52$, the area of the parallelogram, and the
right map has $\Delta = 1\cdot 1 - 2 \cdot \tfrac12 = 0$, a collapsed square.
$|\Delta|$ is the factor by which the map scales area. Chapter 6 gives it its
name, the determinant, and extends it to bigger matrices.

<!-- worked: matrices.inverse_2x2 -->
A = [[4, 7], [2, 6]]: ad - bc = 4*6 - 7*2 = 10.

A^-1 = (1/10) [[6, -7], [-2, 4]] = [[3/5, -7/10], [-1/5, 2/5]].

| check | value |
|---|---:|
| A A^-1 | [[1, 0], [0, 1]] |
| A^-1 A | [[1, 0], [0, 1]] |
| I A | [[4, 7], [2, 6]] |
| x = A^-1 (1, 2) | (-4/5, 3/5) |
| A x | (1, 2) |

Inverse of a product: with B = [[1, 1], [0, 1]] and B^-1 = [[1, -1], [0, 1]], (B A) times A^-1 B^-1 is [[1, 0], [0, 1]], but (B A) times B^-1 A^-1 is [[11/5, -7/5], [2/5, 1/5]].
<!-- /worked -->

The formula is exact for $2 \times 2$ and unusable beyond it: the elimination of
the unknowns generalises, and the general algorithm is the next section.

## Solving systems: Gaussian elimination

Solving $\mathbf{A}\mathbf{x} = \mathbf{y}$ for a large square $\mathbf{A}$ is the
same problem as the $2 \times 2$ case, done systematically. Written out the
system is $n$ equations in $n$ unknowns; the $2 \times 2$ derivation removed an
unknown from an equation by subtracting a multiple of another one. **Gaussian
elimination** does that repeatedly, in an order that leaves a triangle.

Three **row operations** change a system without changing its set of
solutions:

1. swap two equations;
2. multiply an equation by a nonzero number;
3. add a multiple of one equation to another.

Each is reversible (swap again; divide; subtract the multiple), so a vector
satisfies the old system exactly when it satisfies the new one. The method
applies them to the **augmented matrix** $[\mathbf{A}\,|\,\mathbf{y}]$, the
table of coefficients with the right-hand side as an extra column, so that each
operation acts on a whole row.

The goal is a staircase. Work on the first column: pick a row whose entry in it
is nonzero (the **pivot**), swap it to the top, and subtract multiples of it
from each row below so that the column becomes zero under the pivot. Then do the
same in the next column with the rows below the first. The result is a
**row echelon form**: each row's first nonzero entry is to the right of the one
above, and every entry under a pivot is zero. The last equation then contains a
single unknown, which we solve; substituting it into the equation above leaves
one unknown there, and so on up: **back substitution**.

Here are the steps on a system of three equations in three unknowns, each
matrix the one after the stated operation. The notation $R2 \leftarrow R2 +
\tfrac32 R1$ reads "replace row 2 by itself plus $\tfrac32$ times row 1" and
rows are counted from 1 here (columns, in the line that follows the steps,
from 0, as in the pseudocode).

<!-- worked: matrices.elimination_trace -->
The system 2x + y - z = 8, -3x - y + 2z = -11, -2x + y + 2z = -3 as an augmented matrix [A | b]:

start
```text
R1  [   2   1  -1 |   8 ]
R2  [  -3  -1   2 | -11 ]
R3  [  -2   1   2 |  -3 ]
```

R2 <- R2 + 3/2 R1
```text
R1  [   2   1  -1 |   8 ]
R2  [   0 1/2 1/2 |   1 ]
R3  [  -2   1   2 |  -3 ]
```

R3 <- R3 + 1 R1
```text
R1  [   2   1  -1 |   8 ]
R2  [   0 1/2 1/2 |   1 ]
R3  [   0   2   1 |   5 ]
```

R3 <- R3 - 4 R2
```text
R1  [   2   1  -1 |   8 ]
R2  [   0 1/2 1/2 |   1 ]
R3  [   0   0  -1 |   1 ]
```

Pivot columns (counted from 0) [0, 1, 2]: three pivots, so the system has a unique solution. Back substitution from the last row up gives x = (2, 3, -1).

Check: A x = (8, -11, -3), equal to b = (8, -11, -3).
<!-- /worked -->

Three pivots, one in every column, so the staircase has no gaps and the system
has exactly one solution. The final answer can be checked by multiplying back,
and the check on the last line shows that it is satisfied exactly (the
computation uses exact fractions, so there is no rounding to excuse a
difference).

The cost of the method is a count we can make. Eliminating below the pivot in
column $c$ touches about $(n - c)$ rows of about $(n - c)$ entries each, and
summing the squares over the columns gives about $\tfrac13 n^3$ multiply-adds,
plus $\tfrac12 n^2$ for back substitution. The table counts exactly, with this
rule: one multiply-add for every execution of the statement
`M[i, j] = M[i, j] - f * M[r, j]` in `eliminate` below (on the augmented matrix
of shape $[n, n+1]$, where `j` runs from the pivot column to the last column,
so it includes the entry being cleared and the right-hand side), and one for
every term `acc - M[i, j] * x[j]` in back substitution, $n(n-1)/2$ in all.
Divisions (`f`, and the final `acc / M[i, i]`) are not counted. For $n = 3$ that
is $8 + 3 = 11$ in elimination plus $3$ in back substitution, $14$.

<!-- worked: matrices.elimination_cost -->
| n | multiply-adds counted | n^3 / 3 | n^3 (one n x n product) |
|---|---:|---:|---:|
| 3 | 14 | 9 | 27 |
| 10 | 420 | 333 | 1,000 |
| 100 | 343,200 | 333,333 | 1,000,000 |
<!-- /worked -->

The counted values follow $n^3/3$ more and more closely as $n$ grows. Solving
$\mathbf{A}\mathbf{x} = \mathbf{y}$ by elimination therefore costs a third of
what *multiplying two matrices of the same size* costs, and far less than
computing $\mathbf{A}^{-1}$ and multiplying: forming the inverse costs about
$n^3$ (it solves $n$ systems, one per unit vector). Elimination is the standard
way to use the inverse's idea without computing it.

## When an inverse fails: rank

Elimination will not always produce a pivot in every column. Take the matrix
whose rows are $(1,2,3)$, $(4,5,6)$ and $(7,8,9)$.

<!-- worked: matrices.singular_example -->
A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]].

start
```text
R1  [ 1 2 3 ]
R2  [ 4 5 6 ]
R3  [ 7 8 9 ]
```

R2 <- R2 - 4 R1
```text
R1  [  1  2  3 ]
R2  [  0 -3 -6 ]
R3  [  7  8  9 ]
```

R3 <- R3 - 7 R1
```text
R1  [   1   2   3 ]
R2  [   0  -3  -6 ]
R3  [   0  -6 -12 ]
```

R3 <- R3 - 2 R2
```text
R1  [  1  2  3 ]
R2  [  0 -3 -6 ]
R3  [  0  0  0 ]
```

Pivot columns (counted from 0) [0, 1]: only 2 pivots, the last row is all zeros, rank = 2.

| product | value |
|---|---:|
| A (1, 1, 1) | (6, 15, 24) |
| A (1, -2, 1) | (0, 0, 0) |
| A ((1, 1, 1) + 2 (1, -2, 1)) = A (3, -3, 3) | (6, 15, 24) |

Reachable right-hand sides are those with b1 - 2 b2 + b3 = 0 (the elimination leaves b3 - 2 b2 + b1 in the last row):

| b | b1 - 2 b2 + b3 | last row after elimination: 0 = ? | outcome |
|---|---:|---:|---|
| (6, 15, 24) | 0 | 0 | infinitely many solutions |
| (1, 2, 4) | 1 | 1 | no solution |
<!-- /worked -->

After clearing the first column the second row is $(0, -3, -6)$ and the
third $(0, -6, -12)$, and the third is exactly twice the second. Clearing the
second column leaves a row of zeros: the third equation was a consequence of
the other two, and carries no new information. Only two columns have a pivot.
That number is the **rank** of the matrix: the number of pivots that
elimination finds. The matrix is $[3, 3]$ but has rank 2, and in the same table:

- $\mathbf{A}(1,-2,1) = \mathbf{0}$: a nonzero input is sent to the zero
  vector. Whole lines of inputs are crushed to a point, so if $\mathbf{x}$
  solves $\mathbf{A}\mathbf{x} = \mathbf{y}$, so does $\mathbf{x} + t(1,-2,1)$
  for every $t$: infinitely many solutions.
- Not every $\mathbf{y}$ is reachable. The last row of the eliminated system
  reads $0 = y_3 - 2y_2 + y_1$, so a right-hand side with $y_1 - 2y_2 + y_3
  \ne 0$ has *no* solution.

Both facts come from the same cause, and the column view names it. The third
column is $2 \times$ the second minus the first, so the outputs
$x_1\mathbf{a}_{:1} + x_2\mathbf{a}_{:2} + x_3\mathbf{a}_{:3}$ are all
combinations of just two columns, which fill a *plane* in three-dimensional
space. The map is squashing space onto that plane, exactly as the right-hand
panel of Figure 1 squashed the plane onto a line.

To make this precise we need two ideas. Columns are **linearly dependent** if
one of them is a linear combination of the others (they carry a redundancy),
and **independent** otherwise. The **rank** of a matrix is the largest number
of independent columns it has. It is the **dimension of the set of outputs
the map can reach**: one independent column reaches a line, two reach a plane,
three a space, and so on; every extra dependent column adds nothing.

Elimination computes that number because of two facts. First, the row
operations do not change which sets of columns are dependent: a combination of
columns equals the zero vector exactly when it solves the homogeneous system
$\mathbf{A}\mathbf{x} = \mathbf{0}$, and the row operations do not change the
solutions of any system, including that one. Second, in the echelon form the
columns with a pivot are plainly independent (each brings a nonzero entry in
a row where all earlier pivot columns are zero), and every other column is a
combination of the pivot columns to its left. Hence the pivot columns are a
largest independent set, and their number, the number of pivots, is the rank.

The same count has a second reading. The nonzero rows of the echelon form are
independent for the same reason, and the row operations only recombine rows, so
the number of independent *rows* is the number of pivots too:

$$
\operatorname{rank}(\mathbf{A}) = \operatorname{rank}(\mathbf{A}^\top)
\le \min(m, n).
$$

The rank can never exceed the smaller dimension: you cannot have more
independent columns than columns, or than the $m$ entries a column holds. A
matrix with rank equal to $\min(m,n)$ has **full rank**. The survey below
computes the rank of several matrices both ways.

<!-- worked: matrices.rank_survey -->
| matrix | shape | rank of A | rank of A^T | min(m, n) |
|---|---|---:|---:|---:|
| I (3 x 3) | 3 x 3 | 3 | 3 | 3 |
| [[1,2,3],[4,5,6],[7,8,9]] | 3 x 3 | 2 | 2 | 3 |
| outer product (1,2,3)(1,2,-1)^T | 3 x 3 | 1 | 1 | 3 |
| [[1,0,2,1],[0,1,1,3]] | 2 x 4 | 2 | 2 | 2 |
| [[1,2],[2,4],[3,6]] | 3 x 2 | 1 | 1 | 2 |
| zero (2 x 3) | 2 x 3 | 0 | 0 | 2 |
<!-- /worked -->

Reading the survey: the identity has full rank and loses nothing. A rank-1
matrix such as the outer product of two vectors has every column a multiple of
one vector, so its outputs lie on a line. A $[2, 4]$ matrix with rank 2 is full
rank but cannot be invertible; it squeezes four dimensions into two, and
because the map has $4 - 2 = 2$ more input directions than output directions,
two independent input directions are crushed to zero. In general the number of
columns equals the rank plus the number of columns without a pivot, the number of
free directions an input can move in without changing the output. All of this
settles the opening question about undoing a layer:

- **Square and full rank.** There is exactly one solution for every right-hand
  side, nothing is lost, the matrix is invertible.
- **Rank less than $n$.** Some input directions are crushed to zero, the map
  cannot be undone, and any solution comes with infinitely many companions
  differing by those directions.
- **Rank less than $m$.** Some outputs are unreachable, and a system with such a
  right-hand side has no solution at all.

Singular matrices are thus exactly the square ones with rank below $n$, and
elimination detects it as a missing pivot (a column in which no row below
offers a nonzero entry).

## A layer is an affine map

Now the opening problem can be stated exactly. A **linear layer** of a neural
network with $d_{\text{in}}$ inputs and $d_{\text{out}}$ outputs computes

$$
\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b},
\qquad
\mathbf{W} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}},\
\mathbf{b} \in \mathbb{R}^{d_{\text{out}}}.
$$

The matrix $\mathbf{W}$, the **weights**, is the linear part: row $i$ is the
weight vector of output $i$. The vector $\mathbf{b}$, the **bias**, is the
shift, which gives every output an offset of its own. The layer is not a
linear map (it fails $f(\mathbf{0}) = \mathbf{0}$ whenever $\mathbf{b} \neq
\mathbf{0}$) but an **affine map**: linear plus a shift, as in the first section.

Three things matter in practice. First, a layer is applied to many examples
at once. With a batch of $B$ examples stored as the rows of a matrix
$\mathbf{X}$ of shape $[B, d_{\text{in}}]$, the whole batch is a single product:
row $b$ of the result is $\mathbf{W}\mathbf{x}_b + \mathbf{b}$, and in matrix form
$\mathbf{Y} = \mathbf{X}\mathbf{W}^\top + \mathbf{1}\mathbf{b}^\top$, where
$\mathbf{1}\mathbf{b}^\top$ means "add $\mathbf{b}$ to every row" (the
explicit broadcast of the book's [pseudocode](../pseudocode.md) conventions).
The transpose appears because examples are rows here, but the weights act on
columns in the single-example formula. Engines usually store the weight as
$[d_{\text{out}}, d_{\text{in}}]$ and compute $\mathbf{X}\mathbf{W}^\top$, which
is what the next block shows.

Second, a bias is not a separate idea. Append a constant 1 to every input,
$\tilde{\mathbf{x}} = (\mathbf{x}, 1)$, and append $\mathbf{b}$ as one more
column of the weights, $\tilde{\mathbf{W}} = [\mathbf{W}\,|\,\mathbf{b}]$. Then
$\tilde{\mathbf{W}}\tilde{\mathbf{x}} = \mathbf{W}\mathbf{x} + \mathbf{b}$: the
shift is a linear map in one dimension more. A layer with a bias is a linear
map in disguise, and every result about linear maps in this chapter applies to
it after that change.

<!-- worked: matrices.layer_batch -->
W = [[2, -1, 1], [1, 3, 2]] [2, 3], b = (1/2, -1), batch X = [[1, 2, 3], [0, 1, -1]] [2, 3].

| example | x | W x | W x + b |
|---|---|---|---:|
| 0 | (1, 2, 3) | (3, 13) | (7/2, 12) |
| 1 | (0, 1, -1) | (-2, 1) | (-3/2, 0) |

Batched: X W^T = [[3, 13], [-2, 1]]; adding b to every row gives Y = [[7/2, 12], [-3/2, 0]] [2, 2], the same two rows.

One matrix instead: [W | b] = [[2, -1, 1, 1/2], [1, 3, 2, -1]] applied to (x, 1) gives [[7/2, 12], [-3/2, 0]], again the same rows.
<!-- /worked -->

Third, and last, the reason the book does not stop at matrices. Stack two such
layers with no function between them. From the composition section,
$\mathbf{B}(\mathbf{A}\mathbf{x}) = (\mathbf{B}\mathbf{A})\mathbf{x}$: two linear
layers are exactly one linear layer with the matrix $\mathbf{B}\mathbf{A}$. A
hundred linear layers are one. Stacking linear layers gains nothing, but factoring does
save: replacing one $[d, d]$ matrix by a product of two thinner ones with a
narrow middle (the hidden size $h$) stores $2 d h$ numbers instead of $d^2$ and
costs $2 d h$ multiply-adds per input instead of $d^2$, and the product has rank
at most $h$, since all its outputs pass through an $h$-dimensional intermediate. With a bend
between the layers (here ReLU) the stack is no longer any single matrix:

<!-- worked: matrices.stacking_layers -->
Two linear layers x -> Ax -> B(Ax), with A = [[1, 2], [0, 1], [3, -1]], B = [[2, 0, 1], [1, 1, -1]], against the single matrix C = BA = [[5, 3], [-2, 4]]:

| x | B(A x) | C x |
|---|---|---|
| (1, 0) | (5, -2) | (5, -2) |
| (0, 1) | (3, 4) | (3, 4) |
| (2, -1) | (7, -8) | (7, -8) |
| (3, 5) | (30, 14) | (30, 14) |

Put a ReLU between them, h(v) = B relu(A v). For x = (1, 0), y = (-1, 0): h(x + y) = (0, 0), but h(x) + h(y) = (5, -2). The stack is no longer a single matrix.

Factored as [1024, h] times [h, 1024] instead of one [1024, 1024] matrix:

| h | entries 2*d*h | entries d*d | ratio | rank of the product |
|---|---:|---:|---:|---:|
| 16 | 32,768 | 1,048,576 | 0.031 | at most 16 |
| 64 | 131,072 | 1,048,576 | 0.125 | at most 64 |
| 256 | 524,288 | 1,048,576 | 0.500 | at most 256 |
| 1024 | 2,097,152 | 1,048,576 | 2.000 | at most 1024 |
<!-- /worked -->

The second table shows the factoring trade: a narrow middle is much cheaper
($h = 16$ stores a thirtieth of the entries) but limited to rank 16, while at
$h = d$ the factorisation costs twice what one matrix does and gains nothing.
Chapter 6 shows how to find the best narrow factorisation of a given matrix, and
Volume III adds the bend and the training. For the rest of this volume the
matrix is a fixed, known linear map.

## The algorithms

The operations above, in the book's [pseudocode](../pseudocode.md): every shape
assumption is a `require`, arrays are 0-indexed and row-major, and nothing is
broadcast without saying so.

```text
function matvec(A: array<real>[m, n], x: array<real>[n]) -> array<real>[m]:
    require A.shape[1] == x.shape[0], "matvec: inner dimensions differ"
    y = zeros(real, [m])
    for i in 0 .. m:
        acc = 0.0
        for j in 0 .. n:
            acc = acc + A[i, j] * x[j]
        y[i] = acc
    return y

function matmul(A: array<real>[m, k], B: array<real>[k, n]) -> array<real>[m, n]:
    require A.shape[1] == B.shape[0], "matmul: inner dimensions differ"
    C = zeros(real, [m, n])
    parallel for i in 0 .. m:
        parallel for j in 0 .. n:
            acc = 0.0
            for l in 0 .. k:
                acc = acc + A[i, l] * B[l, j]
            C[i, j] = acc
    return C

function transpose(A: array<real>[m, n]) -> array<real>[n, m]:
    T = zeros(real, [n, m])
    for i in 0 .. m:
        for j in 0 .. n:
            T[j, i] = A[i, j]
    return T
```

The iterations of the two outer loops in `matmul` are independent (each writes
only `C[i, j]`), which is why a machine can run all $m n$ dot products at once
and why the work is $m n k$. Elimination, the pivot search and the rank:

```text
function eliminate(M: inout array<real>[m, n], ncols: int, partial: bool) -> array<int>:
    # Forward elimination on the first ncols columns; the remaining columns are
    # an augmented right-hand side and are carried along by every row operation.
    # Returns the pivot columns; their number is the rank of the leading block.
    # The pivot of a column is its first nonzero entry at or below row r, or,
    # when partial is true, the entry of largest absolute value there.
    require 0 <= ncols and ncols <= n
    pivots = []
    r = 0                                   # next row that wants a pivot
    for c in 0 .. ncols:
        if r == m: break
        p = r
        if partial:
            for i in r + 1 .. m:
                if abs(M[i, c]) > abs(M[p, c]): p = i
        else:
            while p < m and M[p, c] == 0: p = p + 1
        if p == m or M[p, c] == 0: continue # no pivot in this column
        swap M[r, :], M[p, :]
        for i in r + 1 .. m:                # clear below the pivot
            f = M[i, c] / M[r, c]
            for j in c .. n:
                M[i, j] = M[i, j] - f * M[r, j]
        append c to pivots
        r = r + 1
    return pivots

function solve(A: array<real>[n, n], y: array<real>[n], partial: bool) -> array<real>[n]:
    M = augment(copy(A), y)                 # [A | y], shape [n, n + 1]
    pivots = eliminate(M, n, partial)
    require len(pivots) == n, "solve: matrix is singular"
    x = zeros(real, [n])
    for i in n-1 down to 0:                 # back substitution
        acc = M[i, n]
        for j in i + 1 .. n:
            acc = acc - M[i, j] * x[j]
        x[i] = acc / M[i, i]
    return x

function rank(A: array<real>[m, n]) -> int:
    return len(eliminate(copy(A), n, false))   # exact arithmetic: first nonzero pivot
```

`solve` refuses a singular matrix instead of dividing by zero, and the pivot
search tests for an exact zero, which is the right test in exact arithmetic and a
dangerous one in floating point; the section below says why and what replaces it.
The loop `for j in c .. n` starts at the pivot column, not at 0, because
everything to its left is already zero.

## Making it work

**Shape errors are bugs, not rounding.** The shape rule is the cheapest check in
all of numerical computing. Every function above `require`s its shapes, and so
should the reader's. A product whose inner dimensions disagree is not
"approximately right": the program has mis-wired two layers, and a silent
truncation or a broadcast that happens to succeed turns an immediate failure
into a model that trains and produces nonsense. The most common instance is the
convention mismatch the layer section named: a weight stored as
$[d_{\text{out}}, d_{\text{in}}]$ needs a transpose in a batched product, a
weight stored as $[d_{\text{in}}, d_{\text{out}}]$ does not, and a square layer
accepts either, so a mix-up goes unnoticed until the sizes differ.

**Group a chain before you run it.** The association-cost table is the
reminder: when a product of several matrices ends in a vector or a small
matrix, evaluate it right to left (or in whatever order keeps intermediates
small), and let the shapes, not the written order, decide. The result is
identical in exact arithmetic; the cost can differ by two orders of magnitude.

**Do not form an inverse to solve a system.** $\mathbf{x} =
\mathbf{A}^{-1}\mathbf{y}$ is the right mathematics and the wrong program:
elimination solves the system for a third of the work, and it does not
compute and then multiply by a matrix that may be almost singular. Compute
an inverse only when the inverse itself is the quantity wanted.

**A zero pivot test is not enough, and a small pivot is a hazard.** In
exact arithmetic any nonzero pivot works. In floating point, dividing by a very
small pivot produces enormous multipliers that swamp the other entries, and the
answer can be wrong while looking fine. The fix, **partial pivoting**, is to
choose, among the rows still available, the one whose entry in the column is
largest in magnitude, and swap it up. The example is a system whose first
pivot is $10^{-17}$:

<!-- worked: matrices.pivot_choice -->
| pivot rule | computed x | largest residual (max over equations of abs(Ax - b)) |
|---|---:|---:|
| first | (0, 1) | 1.0e+00 |
| partial | (1, 1) | 0.0e+00 |

System: 1e-17 x1 + x2 = 1 and x1 + x2 = 2, in 64-bit floating point. The exact solution is x1 = 1/(1 - 1e-17), x2 = 1 - 1e-17 x1, both within 1e-16 of 1.
<!-- /worked -->

Taking the first nonzero pivot divides by $10^{-17}$, loses the second equation
entirely and returns $x_1 = 0$, which violates the second equation by a full
unit (residual 1). Taking the larger pivot gives the right answer. Chapter 15
explains the rounding behind this and why partial pivoting works. The same
fragility affects **rank** on real data, where "exactly zero" almost never
happens: a matrix whose third column is the sum of
two others plus noise of size $10^{-12}$ has rank 3 on paper and, for every
practical purpose, rank 2. Any code that needs a rank must say how small is
small, and Chapter 6 gives a quantity (the singular values) that makes the
question precise.

**The transpose is usually free.** Transposing a stored matrix is, in the
pseudocode above, a copy, but a program rarely needs the copy: reading
$\mathbf{A}^\top$ is reading $\mathbf{A}$ with the two indices swapped, a
different traversal of the same numbers. Chapter 5 makes this exact with
strides; until then, note that a good kernel multiplies by a transposed weight
without ever materialising it.

## Where this appears in brain

The operation of this chapter is, by volume of executed code, most of what
brain does. Four places show it, and one story shows what ignoring its costs.

**The linear layer as a kernel.** `crates/kernels/wgsl/matmul.wgsl` is the
reference implementation of $\mathbf{y} = \mathbf{x}\mathbf{W}^\top$: its header
states that `x` is `[M, K]`, `W` is `[N, K]` (one row per output feature) and
`out[m, n] = sum_k x[m, k] * W[n, k]`, the batched row view with the
convention of the layer section. Each output element is one thread running a
serial loop over $K$, the `acc = acc + x[x_base + i] * w[wi]` line being the
inner loop of the `matmul` pseudocode above. The same directory holds dozens of
variants (`matmul_reg3.wgsl` and its relatives, `matmul_gemv.wgsl`,
`matmul_i8.wgsl`, `matmul_q4_gemv.wgsl`) that compute the same sum and differ in
how the $m n k$ multiply-adds are scheduled and in what number format the weights are stored.
The bias is a separate, cheap step: `crates/model/src/rowemit.rs` has a
`linear` method whose documentation says `y = x @ W^T (+ bias when given)`, which
dispatches the matrix product and then, if there is a bias, a second
kernel that adds it to every row, the broadcast written explicitly. For one vector the host helper `matvec` in
`crates/model/src/hostmath.rs` is the matrix-vector product, documented as
`y = x·W^T` with `W: [out, inn]`, which is the layout of the kernel at $m = 1$.

**The transpose in the backward pass.** `crates/kernels/wgsl/matmul_dx.wgsl`
states in its header the backward of $\mathbf{x}\mathbf{W}^\top$ with respect to
$\mathbf{x}$: `dX[m, k] = sum_n dY[m, n] * W[n, k]`. That is $\mathbf{d}\mathbf{Y}$
times $\mathbf{W}$ with no transpose in the formula because $\mathbf{W}$ is
stored in the transposed layout already, exactly the move the adjoint identity
$(\mathbf{M}\mathbf{x})\cdot\mathbf{y} = \mathbf{x}\cdot(\mathbf{M}^\top\mathbf{y})$
makes, and `crates/kernels/wgsl/matmul_dw.wgsl` computes the matching gradient of the
weights. Volume III derives them; here it is enough to see that a layer's
backward is two more matrix products on the same shapes.

**A test that grouping is not allowed to change results.**
`crates/model/tests/matmul_rows.rs` checks the row-blocked kernel
`matmul_rows.wgsl` against the naive `matmul` and demands bitwise equality over
ragged shapes such as $[13, 32] \times [32, 20]$ and $[17, 5] \times [5, 1]$.
It is a test of a stronger claim than associativity: the row-blocked kernel
changes which thread handles which rows, but keeps the order of the $K$
additions for each output, so floating-point rounding is identical too.
(Reordering the additions would be allowed in exact arithmetic and is not in
floating point, which Chapter 15 explains.)

**A story: the same product, a different schedule.** Commit `c54253ae6`
(20 July 2026) added that row-blocked kernel. Its message explains the
reason in the vocabulary of this chapter: the naive kernel streams the whole
$[N, K]$ weight once for every output row, so for the large vision-transformer
linears of the model in question the weight is read about as many times as
there are rows, and computing eight output rows per thread reads each weight row
once per eight rows. The arithmetic is untouched (the same $m n k$ multiply-adds)
and what changed is the data movement. The cost rule of this chapter counts
arithmetic only, and for large matrices on real hardware memory traffic can
dominate the count, which is a reason Volume IV exists.

A second commit shows the other side of the same lesson: an optimized kernel
that existed but was never reached. Commit `3d438068a` (6 August 2026) found
that a serving engine's prefill dispatched the naive one-thread-per-output
matmul because its kernel table did not contain the register-tiled one, so that
every chunk above 32 rows ran the slow path. The message records the
measurement, a dated observation on one GPU (a P40) with a 0.6B-parameter model:

On 6 August 2026, with a 128-row prefill on one P40 GPU, the matrix product
took 2400.38 ms (98.0% of the pass) in the naive kernel and 78.47 ms with the <!-- perf-number: historical measurement, commit 3d438068a -->
register-tiled `matmul_reg3`, and the whole pass fell from 2207.90 ms to about <!-- perf-number: historical measurement, commit 3d438068a -->
75 to 82 ms. The commit stresses that no new kernel was written: the fix was <!-- perf-number: historical measurement, commit 3d438068a -->
registering and selecting the one that already existed, by the rule
`block::gemm_variant` that the other image models already shared. 

Knowledge #15 records a sibling failure: a selection guard (`m < 128 || n < 128`)
sent short batches to the naive kernel, and the cure was lowering the guard, with
bit-identical output at the same seed. 

The operation is the same matrix product throughout, and its cost is decided
by shapes and schedule: when the $m n k$ count says a product should be cheap
and the profile says it is not, the first question is which kernel ran.

## Check your understanding

Answers are at the end of the chapter.

1. **Predict.** For $\mathbf{A} = \begin{pmatrix}1&2\\3&4\end{pmatrix}$ and $\mathbf{x} = (1, 1)$, compute $\mathbf{A}\mathbf{x}$ twice: once as dot products of rows with $\mathbf{x}$, once as a weighted sum of columns.
2. **Find the error.** "A $[2, 3]$ matrix times another $[2, 3]$ matrix is a $[2, 3]$ matrix." Which rule is broken, and what shape would make the product defined?
3. **Predict.** For $\mathbf{A} = \begin{pmatrix}1&1\\0&1\end{pmatrix}$ and $\mathbf{B} = \begin{pmatrix}1&0\\1&1\end{pmatrix}$, compute $\mathbf{A}\mathbf{B}$ and $\mathbf{B}\mathbf{A}$.
4. **Explain.** $\mathbf{A}$ and $\mathbf{B}$ are $512 \times 512$ and $\mathbf{x}$ has 512 entries. Both $(\mathbf{A}\mathbf{B})\mathbf{x}$ and $\mathbf{A}(\mathbf{B}\mathbf{x})$ give the same vector. Which is cheaper, and by roughly what factor?
5. **Find the error.** "$\begin{pmatrix}1&2\\2&4\end{pmatrix}$ has the inverse $\frac{1}{ad-bc}\begin{pmatrix}d&-b\\-c&a\end{pmatrix}$." Evaluate $ad - bc$ and say what the rank is and what happens to the input directions.
6. **Explain.** A network applies $\mathbf{y} = \mathbf{W}_2(\mathbf{W}_1\mathbf{x})$ with nothing between the layers. Why is it no more expressive than one layer?
7. **Predict.** If $\mathbf{A}$ has shape $[2, 3]$ and $\mathbf{B}$ has shape $[3, 4]$, what are the shapes of $\mathbf{A}\mathbf{B}$ and of $(\mathbf{A}\mathbf{B})^\top$, and what is $(\mathbf{A}\mathbf{B})^\top$ in terms of $\mathbf{A}^\top$ and $\mathbf{B}^\top$?

## Exercises

1. **[Calculate]** For each function from $\mathbb{R}^2$ to $\mathbb{R}^2$ decide
   whether it is linear and prove it from the definition or give a
   counterexample: $f(x_1,x_2) = (x_1 + x_2, x_1 - x_2)$;
   $f(x_1,x_2) = (x_1 x_2, x_2)$; $f(x_1,x_2) = (x_1 + 1, x_2)$;
   $f(x_1,x_2) = (|x_1|, x_2)$. For each linear one write its matrix.

2. **[Calculate]** For $\mathbf{A} = \begin{pmatrix} 1 & 0 & 2 \\ -1 & 3 & 1
   \end{pmatrix}$ and $\mathbf{x} = (2, -1, 4)$ compute $\mathbf{A}\mathbf{x}$ by
   rows and by columns and confirm they agree. What is the output when
   $\mathbf{x} = \mathbf{e}_2$, and what does that tell you about column 2?

3. **[Calculate]** Find the matrices of these maps of the plane from the images
   of $\mathbf{e}_1$ and $\mathbf{e}_2$: the rotation by $90^\circ$ anticlockwise;
   the reflection in the line $x_2 = x_1$; the projection of Chapter 3 onto the
   line through $\mathbf{u} = (1, 1)$. Which of them have an inverse, and what
   is the inverse of each that does?

4. **[Calculate]** With $\mathbf{A} = \begin{pmatrix} 1 & 2 \\ 0 & 1 \end{pmatrix}$
   and $\mathbf{B} = \begin{pmatrix} 2 & 0 \\ 1 & 1 \end{pmatrix}$ compute
   $\mathbf{A}\mathbf{B}$ and $\mathbf{B}\mathbf{A}$, then
   $(\mathbf{A}\mathbf{B})^\top$ and $\mathbf{B}^\top\mathbf{A}^\top$. Verify
   $(\mathbf{A}\mathbf{B})^{-1} = \mathbf{B}^{-1}\mathbf{A}^{-1}$ with the
   $2\times 2$ formula. Prove $(\mathbf{A}\mathbf{B}\mathbf{C})^\top =
   \mathbf{C}^\top\mathbf{B}^\top\mathbf{A}^\top$ from the two-factor rule.

5. **[Calculate]** Four matrices have shapes $[20, 5]$, $[5, 40]$, $[40, 10]$ and
   $[10, 30]$. Count the multiply-adds for the left-to-right order
   $((\mathbf{A}\mathbf{B})\mathbf{C})\mathbf{D}$ and the right-to-left order.
   Then find the cheapest of all five groupings. State a rule for a chain
   ending in a vector.

6. **[Calculate]** Solve by hand with elimination, using a row swap where a
   zero pivot appears, and state the rank:
   $x_2 + 2x_3 = 3$, $2x_1 + x_2 + x_3 = 5$, $x_1 + x_2 + x_3 = 3$.
   Then change the right-hand side of the first equation so that the system has
   no solution, and the third equation so that it has infinitely many.

7. **[Calculate]** For $\mathbf{A} = \begin{pmatrix} 3 & 6 \\ 1 & 2 \end{pmatrix}$
   compute $ad - bc$, say what the map does to the unit square, find a nonzero
   $\mathbf{x}$ with $\mathbf{A}\mathbf{x} = \mathbf{0}$, and give the set of
   right-hand sides $\mathbf{y}$ for which $\mathbf{A}\mathbf{x} = \mathbf{y}$
   is solvable.

8. **[Calculate]** A layer maps $d_{\text{in}} = 8$ to $d_{\text{out}} = 5$ with a
   bias. How many parameters has it? Write the single matrix of shape $[5, 9]$
   that does the same job on $(\mathbf{x}, 1)$, and the shape of $\mathbf{X}
   \mathbf{W}^\top$ for a batch of 32. What is the largest rank
   $\mathbf{W}$ can have?

9. **[Implement]** Implement `matvec` (row view and column view), `matmul` with a
   multiply-add counter and `transpose` from the pseudocode, with errors for
   mismatched shapes, and reproduce `matrices.matvec_two_views`,
   `matrices.composition_is_product`, `matrices.shapes_and_cost` and
   `matrices.association_costs`.

10. **[Implement]** Implement `eliminate`, `solve` and `rank` over exact
    fractions and reproduce `matrices.elimination_trace`,
    `matrices.singular_example`, `matrices.rank_survey` and
    `matrices.elimination_cost` (count as the text's rule says; $n = 10$ and
    $n = 100$ need only the loop structure, so run the counting on floating
    numbers). Add a random test: for 50 random $4\times 4$ integer matrices
    (entries uniform in $-6 \ldots 6$, from any generator, redrawn until the
    rank is 4) and integer right-hand sides, check $\mathbf{A}\,\mathrm{solve}(
    \mathbf{A},\mathbf{y}) = \mathbf{y}$ exactly.

11. **[Implement]** Reproduce `matrices.pivot_choice` in 64-bit floating point with
    first-nonzero pivoting and with partial pivoting (the `partial` argument
    of `eliminate`; the largest residual is the largest over the two equations
    of $\lvert (\mathbf{A}\mathbf{x} - \mathbf{b})_i \rvert$). Then vary the small pivot
    over $10^{-1}, 10^{-5}, 10^{-10}, 10^{-17}$ and report where the unpivoted
    answer first stops matching the pivoted one to six significant digits (each
    component within a relative $5\times10^{-7}$). Expect $10^{-17}$; at
    $10^{-10}$ the two still agree to six digits but the unpivoted $x_1$ is off
    by about $8\times10^{-8}$, so a stricter criterion of eight digits would
    already fail there.

12. **[Investigate]** The projection of Chapter 3 onto the line through
    $\mathbf{u}$ is a linear map with the matrix $\mathbf{P} =
    \mathbf{u}\mathbf{u}^\top / (\mathbf{u}\cdot\mathbf{u})$. Show that it has
    rank 1, that $\mathbf{P}^2 = \mathbf{P}$ (project twice, nothing changes),
    that $\mathbf{P}^\top = \mathbf{P}$, and explain in the column view why it
    cannot have an inverse. Check numerically for $\mathbf{u} = (1, 2, 2)$.

## Summary

A **matrix** is a table of numbers with $m$ rows and $n$ columns, and the same
object is a **linear map** $\mathbb{R}^n \to \mathbb{R}^m$: a function that
commutes with addition and scaling, sends $\mathbf{0}$ to $\mathbf{0}$, and is
completely determined by its action on the unit vectors, which are its columns.
The **matrix-vector product** has two readings of one formula: a list of $m$ dot
products of the rows with the input (the view for computing and for how one
output depends on the input) and a weighted sum of the columns (the view for what
the map can produce). **Matrix multiplication** is not a convention but the
composition of linear maps written in entries, $(\mathbf{B}\mathbf{A})_{ij} =
\sum_l B_{il}A_{lj}$, with the shape rule $[m,k]\times[k,n] = [m,n]$ and a cost of
$m n k$ multiply-adds. It is associative, distributive and not commutative, and
because the grouping of a chain changes its cost but not its value, the order of
evaluation is a first-order decision: a factor of 256 for $\mathbf{A}\mathbf{B}
\mathbf{x}$ at size 512. The **transpose** swaps rows and columns, satisfies
$(\mathbf{A}\mathbf{B})^\top = \mathbf{B}^\top\mathbf{A}^\top$ and moves a map
across a dot product. The **identity** leaves vectors alone; the **inverse**
undoes a map, reverses the order of a product, and for $2\times 2$ is
$\frac{1}{ad-bc}$ times a rearrangement of the entries, failing exactly when
$ad - bc = 0$. **Gaussian elimination** solves a system in about $n^3/3$
multiply-adds and, by finding pivots, reveals the **rank**: the number of
independent columns, equal to the number of independent rows and to the
dimension of what the map can reach. A square matrix with rank below its size
crushes some directions to zero, cannot be inverted, and leaves some outputs
unreachable. A neural network layer $\mathbf{y} = \mathbf{W}\mathbf{x} +
\mathbf{b}$ is an affine map, a linear map in disguise through an appended 1,
and a stack of such layers with nothing in between is one matrix, which is why
the functions between layers carry the expressive power.

What we have not done is say where the numbers live. A matrix has been a table,
and a batch of matrices a pile of tables, but a machine stores numbers in one
long line of memory, and the cost of reading them in the wrong order can swamp
the count of multiply-adds. The next chapter shows how vectors, matrices and
tensors with more axes are laid out, how an index becomes an address, why a
transpose can be free, and how arrays of different shapes are combined without
copying.

## Key terms

- **matrix**: a table of numbers with $m$ rows and $n$ columns, which is also a linear map from $n$ numbers to $m$ numbers. <!-- concept: matrix -->
- **linear map**: a function of vectors that commutes with addition and scaling, and is determined by what it does to the unit vectors. <!-- concept: linear-map -->
- **matrix-vector product**: the vector whose entries are the dot products of the rows of a matrix with the input vector, equivalently a weighted sum of the matrix's columns. <!-- concept: matrix-vector-product -->
- **matrix multiplication**: the matrix of the composition of two linear maps, $(\mathbf{B}\mathbf{A})_{ij} = \sum_l B_{il}A_{lj}$, defined when the inner sizes agree. <!-- concept: matrix-multiplication -->
- **transpose**: the matrix $\mathbf{A}^\top$ with rows and columns swapped; $(\mathbf{A}\mathbf{B})^\top = \mathbf{B}^\top\mathbf{A}^\top$. <!-- concept: transpose -->
- **identity matrix**: the square matrix with ones on the diagonal and zeros elsewhere, which leaves every vector unchanged. <!-- concept: identity-matrix -->
- **matrix inverse**: the matrix $\mathbf{A}^{-1}$ that undoes $\mathbf{A}$, so $\mathbf{A}^{-1}\mathbf{A}$ is the identity; it exists only for a square matrix whose rank equals its size. <!-- concept: matrix-inverse -->
- **rank**: the number of independent columns of a matrix, equal to the number of independent rows and to the dimension of what the map can reach. <!-- concept: rank -->

## Answers

**Before you begin.**

1. $4 + 10 + 18 = 32$.
2. The vectors are orthogonal: at right angles, with none of one pointing along the other.
3. $g(f(2)) = g(5) = 25$; $f$ is applied first, then $g$.
4. No: a function gives exactly one output for each input. Yes: two inputs may share an output (for example $x^2$ at $2$ and $-2$); that is what stops such a function from having an inverse.
5. $1\cdot 2 + 2 \cdot 3 + 3 \cdot 4 = 2 + 6 + 12 = 20$.

**Check your understanding.**

1. Rows: $(1\cdot 1 + 2\cdot 1,\ 3\cdot 1 + 4\cdot 1) = (3, 7)$. Columns: $1\cdot(1, 3) + 1\cdot(2, 4) = (3, 7)$.
2. The inner sizes must agree: $[m, k] \times [k, n] = [m, n]$, and here the inner sizes are 3 and 2. A $[3, n]$ second factor makes it defined, giving $[2, n]$.
3. $\mathbf{A}\mathbf{B} = \begin{pmatrix}2&1\\1&1\end{pmatrix}$ and $\mathbf{B}\mathbf{A} = \begin{pmatrix}1&1\\1&2\end{pmatrix}$. They differ: matrix multiplication is not commutative.
4. $\mathbf{A}(\mathbf{B}\mathbf{x})$ costs two matrix-vector products, $2 \cdot 512^2$ multiply-adds. $(\mathbf{A}\mathbf{B})\mathbf{x}$ first forms a matrix product at $512^3$ multiply-adds, then one more matrix-vector product. The ratio is about $513/2$, roughly 256 times more.
5. $ad - bc = 1\cdot 4 - 2\cdot 2 = 0$, so the formula divides by zero and there is no inverse. The second row is twice the first, the rank is 1, and the map crushes a whole direction to zero, so the original input cannot be recovered.
6. $\mathbf{W}_2(\mathbf{W}_1\mathbf{x}) = (\mathbf{W}_2\mathbf{W}_1)\mathbf{x}$ by associativity, so the two layers are one matrix. The functions between layers are what add expressive power.
7. $\mathbf{A}\mathbf{B}$ is $[2, 4]$ and $(\mathbf{A}\mathbf{B})^\top$ is $[4, 2]$; it equals $\mathbf{B}^\top\mathbf{A}^\top$ (shapes $[4, 3]$ and $[3, 2]$), with the order reversed.
