<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# The Backward Pass of Every Layer

*What is the backward pass of a matrix multiply, a sum, a broadcast and an elementwise function, written out exactly?*

## The problem

The previous chapter derived the backward pass of one small network, two linear layers around a $\tanh$, and then stated a design: every operation in an engine supplies a rule that takes the gradient arriving from above and returns the gradients going down. That is a contract, not a rule. A real network is built from many more operations than the ones we differentiated: matrix products with a leading batch axis, sums and means, maximums, softmax, additions of a small array to a large one, reshapes, slices, lookups by index, and weights that are used twice. Each of them needs its backward written exactly, once, so that nobody ever has to derive it again under pressure.

Start with the smallest case. A bias $\mathbf{b} = (0.5, -0.5, 0, 1)$ is added to each of three rows of an activation. The loss sends back a gradient for every entry of the result, and in column 1 the three rows hold $-1$, $2$ and $1$. What is the gradient of $b_1$? Chapter 4's fan-out rule answers it: $b_1$ was used three times, once per row, each use passes its gradient straight through (the derivative of an addition is $1$), and the contributions add: $-1 + 2 + 1 = 2$. A one-line rule hides in that sentence, *a value that was copied forward gets the sum of the gradients of its copies*, and this chapter shows that every layer's backward pass is one such statement about *who read what*. We derive them all with one method, check every one numerically, count what each costs, and end with the rule that keeps them from silently going wrong: gradients **accumulate**.

You may use from earlier chapters: the backward pass, the vector-Jacobian product and the fan-out rule (Chapter 4), the softmax Jacobian (Chapter 3), matrix products and transposes (Volume I, Chapter 4), broadcasting (Volume I, Chapter 5) and the multivariable chain rule (Volume I, Chapter 8).

## Before you begin

Five questions on ideas from earlier chapters, to answer from memory. The answers are at the end of the chapter. If one is hard, go back to the chapter named before reading on.

1. ("Gradients, Jacobians and Many Variables", Volume I, Chapter 8, the chain rule.) A quantity $z$ depends on $x$ through three intermediate quantities $y_1(x), y_2(x), y_3(x)$. Write $\frac{dz}{dx}$ in terms of partial derivatives.
2. ("Backpropagation Derived from Scratch", Chapter 4 of this volume.) For $a_i = \tanh(z_i)$ the backward pass computes $dz_i = da_i\,(1 - a_i^2)$ without building a Jacobian. What shape does the Jacobian have, and why can the product be done with $n$ multiplications instead of $n^2$?
3. ("Backpropagation Derived from Scratch".) A linear layer maps $\mathbf{X}$ of shape $[B, D]$ with a weight $\mathbf{W}$ of shape $[O, D]$ to $\mathbf{Y} = \mathbf{X}\mathbf{W}^\top$. What are the shapes of $d\mathbf{Y}$, $d\mathbf{W}$ and $d\mathbf{X}$?
4. ("Arrays, Shapes, Strides and Broadcasting", Volume I, Chapter 5.) What does it mean to add an array $\mathbf{b}$ of shape $[4]$ to an array of shape $[3, 4]$? Which axis is stretched, and does the stretching copy any memory?
5. ("Matrices as Linear Maps", Volume I, Chapter 4.) If $\mathbf{A}$ is $[m, k]$ and $\mathbf{B}$ is $[k, n]$, what is $(\mathbf{A}\mathbf{B})^\top$ in terms of $\mathbf{A}^\top$ and $\mathbf{B}^\top$?

## One method for every rule

Chapter 4 derived each backward formula by looking at one entry. We make that the method. Suppose an operation computes outputs $y_j$ from inputs $x_i$, and the gradient $dy_j = \partial L / \partial y_j$ of the loss with respect to every output is known. The multivariable chain rule says the loss reaches $x_i$ through every output that read $x_i$, and the paths add:

$$
dx_i = \sum_j dy_j \,\frac{\partial y_j}{\partial x_i}. \tag{1}
$$

That is the vector-Jacobian product of Chapter 4 written one entry at a time, and it is the only formula in this chapter. What changes from operation to operation is which $j$ appear in the sum and what the partial derivative is. In nearly every layer of a network the partial derivative is trivial (a weight, an input, a constant), so the work is bookkeeping, and the bookkeeping has one rule:

> **Every place the forward pass reads a value becomes a place where the backward pass adds into that value's gradient.**

A forward read of $x_i$ by output $y_j$ becomes a term $dy_j \cdot \partial y_j/\partial x_i$ added into $dx_i$. If an element is read once, its gradient is one term. If it is read $B$ times (a bias, a weight shared by a batch), its gradient is a sum of $B$ terms. If it is never read (an element sliced away), its gradient is zero. The recipe, then, has three steps. Write one output entry with explicit indices. Find every output entry that mentions the input entry you are differentiating. Add the products, and notice the matrix operation the sums form.

## A linear layer with every number

A **batched linear layer** computes $\mathbf{Y} = \mathbf{X}\mathbf{W}^\top + \mathbf{1}\mathbf{b}^\top$ for a batch $\mathbf{X}$ of $B$ examples (one per row). Written with indices, one output entry is

$$
Y_{bo} = \sum_{d} X_{bd}\, W_{od} + b_o .
$$

Here is a layer small enough to follow by hand, with all three sizes different (a transposed matrix then becomes a shape error instead of a quietly wrong answer; Chapter 4 explained why). The numbers in $d\mathbf{Y}$ are arbitrary: they stand for whatever the rest of the network sends back, and we define the loss as $L = \sum_{b,o} dY_{bo}\,Y_{bo}$ so that the gradient arriving at $\mathbf{Y}$ is exactly $d\mathbf{Y}$.

<!-- worked: layer_gradients.linear_inputs -->
A linear layer with B = 3 examples, D = 2 inputs and O = 4 outputs, `Y = X W^T + b`.

| X [B, D] | d=0 | d=1 |
|---|---:|---:|
| b=0 | 1.0 | 2.0 |
| b=1 | 0.0 | -1.0 |
| b=2 | 3.0 | 1.0 |

| W [O, D] | d=0 | d=1 |
|---|---:|---:|
| o=0 | 1.0 | 0.0 |
| o=1 | 0.0 | 1.0 |
| o=2 | 1.0 | 1.0 |
| o=3 | 2.0 | -1.0 |

| b [O] | o=0 | o=1 | o=2 | o=3 |
|---|---:|---:|---:|---:|
| value | 0.5 | -0.5 | 0.0 | 1.0 |

The loss is `L = sum over b, o of G[b, o] * Y[b, o]`, so the gradient arriving at `Y` is `dY = G`:

| dY = G [B, O] | o=0 | o=1 | o=2 | o=3 |
|---|---:|---:|---:|---:|
| b=0 | 1.0 | -1.0 | 2.0 | 0.0 |
| b=1 | 0.0 | 2.0 | -1.0 | 1.0 |
| b=2 | -2.0 | 1.0 | 0.0 | 3.0 |
<!-- /worked -->

The forward pass gives:

<!-- worked: layer_gradients.linear_forward -->
| Y = X W^T + b [B, O] | o=0 | o=1 | o=2 | o=3 |
|---|---:|---:|---:|---:|
| b=0 | 1.5 | 1.5 | 3.0 | 1.0 |
| b=1 | 0.5 | -1.5 | -1.0 | 2.0 |
| b=2 | 3.5 | 0.5 | 4.0 | 6.0 |

The loss is L = 17.5.
<!-- /worked -->

Now apply the recipe to each input of the layer. Take a weight entry $W_{od}$. It is read by every output in column $o$, one per example, so by equation (1)

$$
dW_{od} = \sum_{b} dY_{bo}\,\frac{\partial Y_{bo}}{\partial W_{od}} = \sum_{b} dY_{bo}\, X_{bd}.
$$

The sum runs over $b$, the index that $dY$ and $X$ share and that $dW$ does not have. A sum over a shared index is a matrix product, here of $d\mathbf{Y}^\top$ (which is $[O, B]$) with $\mathbf{X}$ (which is $[B, D]$). An input entry $X_{bd}$ is read by every output in row $b$, one per output unit:

$$
dX_{bd} = \sum_{o} dY_{bo}\, W_{od}, \qquad\text{a product of } d\mathbf{Y}\ [B, O] \text{ and } \mathbf{W}\ [O, D].
$$

A bias entry $b_o$ is read by every row of column $o$, with derivative $1$:

$$
db_o = \sum_{b} dY_{bo}.
$$

Together:

$$
\boxed{\;d\mathbf{X} = d\mathbf{Y}\,\mathbf{W}, \qquad d\mathbf{W} = d\mathbf{Y}^\top\mathbf{X}, \qquad d\mathbf{b} = \sum_b d\mathbf{Y}_{b,\cdot}\;}
$$

These three are the **backward pass of a linear layer**. Chapter 4 derived them inside a network; here they stand alone, and they are the rules that every later layer of this book is built on. A mnemonic is the shapes. The gradient of an array always has that array's shape: $d\mathbf{X}$ is $[B, D]$, $d\mathbf{W}$ is $[O, D]$, $d\mathbf{b}$ is $[O]$. To produce an $[B, D]$ from $d\mathbf{Y}$ $[B, O]$ and the one other array that mentions $D$, namely $\mathbf{W}$ $[O, D]$, there is only one product with matching inner dimensions, $d\mathbf{Y}\mathbf{W}$. To produce an $[O, D]$ from $d\mathbf{Y}$ and $\mathbf{X}$ $[B, D]$ the shared index to sum over is $B$, so $d\mathbf{Y}$ must be transposed first. If you forget a formula, the shapes rebuild it.

![The backward pass of one linear layer. Top: the forward pass from X of shape 3 by 2 to Y of shape 3 by 4. Bottom: the gradient dY of shape 3 by 4 feeds three products, each written with its shapes: dX equals dY times W, dW equals dY transposed times X, and db is dY summed over the batch axis. Each result has the shape of the array whose gradient it is.](img/linear-layer-backward.svg)

On the numbers:

<!-- worked: layer_gradients.linear_backward -->
| dX = dY W [B, D] | d=0 | d=1 |
|---|---:|---:|
| b=0 | 3.0 | 1.0 |
| b=1 | 1.0 | 0.0 |
| b=2 | 4.0 | -2.0 |

| dW = dY^T X [O, D] | d=0 | d=1 |
|---|---:|---:|
| o=0 | -5.0 | 0.0 |
| o=1 | 2.0 | -3.0 |
| o=2 | 2.0 | 5.0 |
| o=3 | 9.0 | 2.0 |

| db = sum over b of dY [O] | o=0 | o=1 | o=2 | o=3 |
|---|---:|---:|---:|---:|
| value | -1.0 | 2.0 | 1.0 | 4.0 |

One entry of each, from the sums over the shared index: dW[2, 1] = sum over b of dY[b, 2] X[b, 1] = (2)(2) + (-1)(-1) + (0)(1) = 5; dX[0, 1] = sum over o of dY[0, o] W[o, 1] = (1)(0) + (-1)(1) + (2)(1) + (0)(-1) = 1.
<!-- /worked -->

Every entry of those tables is a sum of the form we derived, and the last paragraph of the block writes two of them out. To be sure the rules are right and not just plausible, compare them with the definition of the derivative. Nudge one entry of $\mathbf{X}$, $\mathbf{W}$ or $\mathbf{b}$ up and down by $10^{-6}$, recompute $L$, and divide the difference of the two losses by $2 \times 10^{-6}$ (a central difference, previewed in Chapter 4 and made rigorous in the next chapter). Doing that for every entry of every input gives a second, independent gradient:

<!-- worked: layer_gradients.linear_check -->
Central difference with step 1e-6 against the three rules, every entry:

| gradient | shape | largest absolute difference |
|---|---|---:|
| dX | [3, 2] | 5.6e-10 |
| dW | [4, 2] | 5.2e-10 |
| db | [4] | 5.6e-10 |
<!-- /worked -->

The two agree to about nine digits. We use this check for every rule below, and a table at the end collects them all.

The three rules are also a pseudocode function. Matching the contract of Chapter 4, it takes the saved input $\mathbf{X}$ and the weights, and *adds* into the parameter gradients it is handed.

```text
function linear_backward(X: array<real>[B, D], W: array<real>[O, D],
                         dY: array<real>[B, O],
                         dW: inout array<real>[O, D], db: inout array<real>[O])
        -> array<real>[B, D]:
    G = matmul(transpose(dY), X)                       # [O, D]
    for o in 0 .. O:
        for d in 0 .. D:
            dW[o, d] = dW[o, d] + G[o, d]              # add, never assign
    s = sum(dY, axis=0)                                # [O]
    for o in 0 .. O:
        db[o] = db[o] + s[o]
    return matmul(dY, W)                               # dX, [B, D]
```

`dW` and `db` are `inout` because they are the persistent gradient buffers of the parameters. The reason they are added into, rather than overwritten, is the last section of this chapter. `dX` is returned fresh because it is the gradient of a value the layer read once.

## Matrix multiplication in general

The linear layer is a matrix product $\mathbf{X}\mathbf{W}^\top$ followed by a broadcast addition. Separate them. For a plain product $\mathbf{C} = \mathbf{A}\mathbf{B}$ with $\mathbf{A}$ of shape $[m, k]$ and $\mathbf{B}$ of shape $[k, n]$, one entry is $C_{ij} = \sum_k A_{ik} B_{kj}$. The entry $A_{ik}$ is read by every output in row $i$ of $\mathbf{C}$ (one per $j$), and $B_{kj}$ by every output in column $j$ (one per $i$), so

$$
dA_{ik} = \sum_j dC_{ij}\, B_{kj}, \qquad dB_{kj} = \sum_i A_{ik}\, dC_{ij}.
$$

The first is the entry of $d\mathbf{C}\,\mathbf{B}^\top$, the second of $\mathbf{A}^\top d\mathbf{C}$:

$$
\boxed{\;d\mathbf{A} = d\mathbf{C}\,\mathbf{B}^\top \quad [m, n]\times[n, k], \qquad d\mathbf{B} = \mathbf{A}^\top d\mathbf{C} \quad [k, m]\times[m, n]\;}
$$

This is the **backward pass of a matrix multiply**. The linear layer is the special case $\mathbf{A} = \mathbf{X}$, $\mathbf{B} = \mathbf{W}^\top$: then $d\mathbf{X} = d\mathbf{Y}(\mathbf{W}^\top)^\top = d\mathbf{Y}\mathbf{W}$ and $d(\mathbf{W}^\top) = \mathbf{X}^\top d\mathbf{Y}$, whose transpose is $d\mathbf{W} = d\mathbf{Y}^\top\mathbf{X}$ (by $(\mathbf{A}\mathbf{B})^\top = \mathbf{B}^\top\mathbf{A}^\top$). The two descriptions agree, as they must.

**Batched products.** Attention and many other layers multiply a stack of matrices: $\mathbf{C}_i = \mathbf{A}_i \mathbf{B}_i$ for each item $i$ of a leading batch axis, with shapes $[N, m, k]$, $[N, k, n]$ and $[N, m, n]$. The items do not interact, so each item's gradient is the plain rule on its own, $d\mathbf{A}_i = d\mathbf{C}_i\mathbf{B}_i^\top$ and $d\mathbf{B}_i = \mathbf{A}_i^\top d\mathbf{C}_i$, and a batched backward is just the batched version of the same two products.

The case that matters most in practice has a *shared* right operand: $\mathbf{C}_i = \mathbf{A}_i\mathbf{B}$, one weight matrix applied to every item of a batch, as in a linear layer applied to a sequence of positions. Now $\mathbf{B}$ is read by every item. By the rule at the top of this chapter its gradient is the sum of one contribution per item:

$$
d\mathbf{A}_i = d\mathbf{C}_i\,\mathbf{B}^\top, \qquad d\mathbf{B} = \sum_{i=0}^{N-1} \mathbf{A}_i^\top d\mathbf{C}_i .
$$

This is exactly the weight gradient of a linear layer in disguise: stacking the items into one tall matrix $\mathbf{A}$ of shape $[N m, k]$ turns the shared-weight product into a single matrix product, and $\mathbf{A}^\top d\mathbf{C}$ for the tall matrices is the same sum over $i$ (the inner dimension of the product now runs over items and rows together).

<!-- worked: layer_gradients.batched_matmul -->
Seed 17980087011661528469. Draw order: every entry is `integer(7) - 3`, so a value in -3 .. 3; first `A` (2 x 2 x 3 = 12 values, batch index slowest), then the shared `B` (3 x 4 = 12 values), then `dC` (2 x 2 x 4 = 16 values).

Forward: `C[i] = A[i] B` for batch item `i`, one weight matrix `B` shared by both items.

- `A[0]` = [[1, -1, 2], [-2, -3, -1]], `dC[0]` = [[1, 3, 1, -3], [-1, -2, 1, 2]]
- `A[1]` = [[-2, 1, -3], [3, 3, -1]], `dC[1]` = [[-2, -2, -3, -2], [-1, -3, 2, 1]]
- `B` = [[1, 1, -3, -2], [-2, 1, -2, -2], [-2, -2, -2, -2]]

| gradient | value |
|---|---|
| dA[0] = dC[0] B^T | [[7, 5, -4], [-10, -6, 0]] |
| dA[1] = dC[1] B^T | [[9, 12, 18], [-12, -7, 2]] |
| A[0]^T dC[0] | [[3, 7, -1, -7], [2, 3, -4, -3], [3, 8, 1, -8]] |
| A[1]^T dC[1] | [[1, -5, 12, 7], [-5, -11, 3, 1], [7, 9, 7, 5]] |
| dB = A[0]^T dC[0] + A[1]^T dC[1] | [[4, 2, 11, 0], [-3, -8, -1, -2], [10, 17, 8, -3]] |

Largest absolute difference from a central difference: dA 2.5e-09, dB 2.5e-09.
<!-- /worked -->

The sum for $d\mathbf{B}$ is visible in the table: the third and fourth rows are the two per-item contributions, and the last row is their sum. This is the first appearance of a pattern the chapter returns to: the shared object, read $N$ times, receives $N$ terms.

## Elementwise functions

An elementwise function applies the same scalar function to every entry, $y_i = f(x_i)$ (the activations of Chapter 2). Output $y_j$ reads only $x_j$, so in equation (1) the sum has one term, $j = i$, and the Jacobian is diagonal:

$$
\boxed{\;dx_i = dy_i\, f'(x_i)\;}
$$

This is the **backward pass of an elementwise function**: an elementwise product of the incoming gradient with the derivative of $f$ at the input, $n$ multiplications for $n$ entries. It adds no new shape, since $d\mathbf{x}$ has the shape of $\mathbf{x}$ and of $d\mathbf{y}$.

What the backward pass must *remember* is the question Chapter 4 raised in general. The rule needs $f'$ evaluated at the input $x_i$. Often that can be computed from the output $y_i$ instead, which may be the value the engine already holds:

<!-- worked: layer_gradients.elementwise_rules -->
Derivatives at x = 0.5.

| function | f(x) | f'(x) from the input | f'(x) from the output | the backward keeps |
|---|---:|---:|---:|---|
| tanh | 0.462117 | 0.786448 | 0.786448 | output y |
| sigmoid | 0.622459 | 0.235004 | 0.235004 | output y |
| ReLU | 0.500000 | 1.000000 | 1.000000 | one bit per entry (y > 0) |
| GELU | 0.345731 | 0.867495 | not a function of y | input x |
| SiLU | 0.311230 | 0.739961 | not a function of y | input x |
<!-- /worked -->

For $\tanh$ and the sigmoid the derivative is a polynomial of the output ($1 - y^2$ and $y(1 - y)$), so saving the output is enough and the input can be discarded. For ReLU the derivative is $1$ where $x > 0$ and $0$ elsewhere, and $y > 0$ says the same thing, so one bit per entry is enough (a convention is needed at exactly $x = 0$, where the function has a kink; Chapter 4 noted that engines usually choose $0$, and every rule table below uses it).

For GELU and SiLU the output does not determine the derivative, and it is worth seeing why rather than accepting it. GELU is not monotonic: it falls to a minimum near $x = -0.75$ and rises on both sides. A given output below zero is therefore produced by two different inputs, with different derivatives:

<!-- worked: layer_gradients.gelu_two_inputs -->
GELU is not monotonic: below its minimum (near x = -0.75) it falls, above it rises. The output -0.1 is produced by two different inputs, found by 60 bisection steps on [-3, -0.75] and [-0.75, 0].

| input x | gelu(x) | gelu'(x) |
|---:|---:|---:|
| -1.501654 | -0.100000 | -0.127415 |
| -0.248940 | -0.100000 | 0.305421 |
<!-- /worked -->

The same output $-0.1$ goes with a derivative of $-0.127$ in one case and $+0.305$ in the other, with opposite signs, so no function of the output alone can give the derivative; the backward pass must read the input. The choice of what to keep is a property of the function, and it decides the memory the backward pass pins in place, which the table near the end of the chapter counts.

A full check on one vector, with the rule written through the output ($1 - y^2$) and through the input ($1 - \tanh^2 x$), and a central difference beside them:

<!-- worked: layer_gradients.elementwise_vjp -->
| i | x | y = tanh(x) | dy | dx = dy * (1 - y^2) | dx from the input | central difference |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | -1 | -0.761594 | 0.5 | 0.209987 | 0.209987 | 0.209987 |
| 1 | 0 | 0.000000 | -1 | -1.000000 | -1.000000 | -1.000000 |
| 2 | 0.5 | 0.462117 | 2 | 1.572895 | 1.572895 | 1.572895 |
| 3 | 2 | 0.964028 | 1 | 0.070651 | 0.070651 | 0.070651 |
<!-- /worked -->

**Operations with two inputs.** Add, subtract, multiply and divide combine two arrays of the same shape elementwise, $z_i = g(x_i, y_i)$. Output $z_i$ reads $x_i$ and $y_i$ only, so each input's gradient is $dz_i$ times that input's own partial derivative:

| operation | $dx_i$ | $dy_i$ |
|---|---|---|
| $z = x + y$ | $dz_i$ | $dz_i$ |
| $z = x - y$ | $dz_i$ | $-dz_i$ |
| $z = x\,y$ | $dz_i\,y_i$ | $dz_i\,x_i$ |
| $z = x / y$ | $dz_i / y_i$ | $-dz_i\,x_i / y_i^2$ |

An addition passes the gradient to both inputs unchanged, which is the fan-out rule in its purest form. A product needs *the other factor*, so a multiplication saves both of its inputs. If the two inputs have different shapes, one is first broadcast to the shape of the other, and the gradient of the small one is summed back, which the broadcasting section below derives.

<!-- worked: layer_gradients.binary_rules -->
Inputs x = [1, -2, 3], y = [2, 4, -1], upstream dz = [1, 0.5, -2].

| operation | dx | dy | largest difference from central difference |
|---|---|---|---:|
| z = x + y | [1, 0.5, -2] | [1, 0.5, -2] | 2.8e-10 |
| z = x - y | [1, 0.5, -2] | [-1, -0.5, 2] | 2.8e-10 |
| z = x * y | [2, 2, 2] | [1, -1, -6] | 4.9e-10 |
| z = x / y | [0.5, 0.125, 2] | [-0.25, 0.0625, 6] | 2.8e-10 |
<!-- /worked -->

## Reductions

A reduction collapses an axis: a sum, a mean, a maximum. The **backward pass of a reduction** is the mirror of the elementwise one. An elementwise output reads one input; a reduction output reads many inputs. Equation (1) then has, for each input, one output to sum over, and the question is only what $\partial y_j / \partial x_i$ is.

**Sum.** For $r = \sum_{i} x_i$ the derivative $\partial r/\partial x_i$ is $1$ for every $i$, so every input receives the same number, the gradient of the result:

$$
dx_i = dr \quad\text{for every } i .
$$

The gradient is *copied* to every input: the backward of a sum is a broadcast. If a matrix is summed along an axis, each row (or column) of the result sends its gradient to all entries it collapsed.

**Mean.** $r = \frac{1}{n}\sum_i x_i$ has $\partial r/\partial x_i = 1/n$, so $dx_i = dr/n$. This $1/n$ is the factor that is easy to drop by accident (Chapter 4 made the same point about the $1/(BO)$ of the mean squared error). A mean loss over $n$ examples gives each example a gradient $n$ times smaller than a sum loss does.

**Maximum.** $r = \max_i x_i$ equals one input, the largest, and is locally independent of the rest: nudging a non-maximal entry does not change $r$. So the gradient goes entirely to the position of the maximum and is zero everywhere else:

$$
dx_i = \begin{cases} dr & \text{if } i = \operatorname{argmax}(\mathbf{x}) \\ 0 & \text{otherwise.}\end{cases}
$$

This is where a rule is needed that mathematics does not give. If two entries tie for the maximum, $\max$ has a kink: moving the first entry up raises $r$, moving it down does not. There is no single derivative. Any assignment that sends the gradient to tied positions in shares summing to $1$ is a valid choice; an engine picks one and sticks to it. The book's `argmax` returns the first index on a tie (pseudocode conventions), so the first-index rule is the one we use, and the table shows what a central difference sees at a tie.

<!-- worked: layer_gradients.reductions -->
Input A = [[1, 3, 2], [4, 4, 0]] (2 rows, 3 columns), reduced along each row; the gradient arriving at the two row results is dr = [2, -1].

| reduction | result per row | dA | largest difference from central difference |
|---|---|---|---:|
| sum | [6, 8] | [[2, 2, 2], [-1, -1, -1]] | 5.8e-10 |
| mean | [2, 2.667] | [[0.6667, 0.6667, 0.6667], [-0.3333, -0.3333, -0.3333]] | 2.7e-10 |
| max, first index wins ties | [3, 4] | [[0, 2, 0], [-1, 0, 0]] | row 0: 2.8e-10; row 1 is tied |

At the tied row the central difference reports [-0.5, -0.5, 0]: it splits dr[1] = -1 evenly between the two tied entries, while the rule above gives all of it to column 0.
<!-- /worked -->

The central difference at the tied row is $-0.5$ at each tied position: it measures the average of the two one-sided slopes. A gradient checker therefore cannot validate a rule at a tie, and test inputs must avoid ties and kinks (the ReLU, $0$ rules, and the max rule alike) by a margin larger than the step. The table of rules at the end of the chapter does that and reports the margin.

```text
function max_backward(X: array<real>[B, C], dr: array<real>[B]) -> array<real>[B, C]:
    dX = zeros(real, [B, C])
    for i in 0 .. B:
        j = argmax(X[i, :])                  # the first index on a tie
        dX[i, j] = dr[i]
    return dX
```

**Softmax, a row-coupled operation.** Softmax is not a sum or a maximum, but it sits here because, like a reduction, each output reads the whole row. Chapter 3 derived its Jacobian, $\mathrm{diag}(\mathbf{y}) - \mathbf{y}\mathbf{y}^\top$, and the product of an upstream vector with it:

$$
\boxed{\;dx_i = y_i\Bigl(dy_i - \sum_j dy_j\, y_j\Bigr)\;}
$$

This is a VJP of exactly the kind this chapter is about: one dot product per row and one elementwise product, with no Jacobian built. It reads the output $\mathbf{y}$, so softmax saves its output. The dot product couples the row and has a visible consequence: the gradients $dx_i$ of one row sum to $\sum_i y_i dy_i - (\sum_i y_i)(\sum_j dy_j y_j) = 0$, because adding a constant to every logit changes nothing.

<!-- worked: layer_gradients.softmax_backward -->
Rows of logits z = [[1, 2, 3], [0, 0, 1]] and upstream dy = [[1, 0, -1], [0.5, 2, 1]].

| y = softmax(z) | c=0 | c=1 | c=2 |
|---|---:|---:|---:|
| b=0 | 0.090031 | 0.244728 | 0.665241 |
| b=1 | 0.211942 | 0.211942 | 0.576117 |

| dot = sum of dy * y, then dx = y * (dy - dot) | c=0 | c=1 | c=2 |
|---|---:|---:|---:|
| b=0 | 0.141817 | 0.140770 | -0.282587 |
| b=1 | -0.128430 | 0.189482 | -0.061052 |

Row sums of dx: -2.8e-17 and 3.5e-17 (a shift of all logits changes nothing). Largest difference from a central difference: 7.5e-11.
<!-- /worked -->

**A duality.** Put the sum and the broadcast side by side. The backward of a sum is a broadcast (copy the gradient to every input it collapsed); the next section shows that the backward of a broadcast is a sum. Reductions and broadcasts are each other's adjoints, and this single pair explains the bias gradient and the $1/n$ of a mean loss, and the normalisation layers of a later chapter are built from it.

## Broadcasting

Volume I, Chapter 5 defined broadcasting: when two arrays of different shapes meet in an elementwise operation, axes of size $1$ (and missing leading axes) are stretched, without copying, to match. The forward pass reads the small array many times. Our rule says exactly what the backward pass does about that.

Take $Y_{bo} = X_{bo} + b_o$, a bias of shape $[O]$ added to every row of an $[B, O]$ array. The entry $b_o$ is read by $B$ outputs, one per row. Each read contributes $dY_{bo}\cdot 1$, and the contributions add:

$$
db_o = \sum_{b} dY_{bo}.
$$

Chapter 4's fan-out rule says the same thing in the language of the graph: a bias used in $B$ places has $B$ outgoing edges, and its gradient is the sum over them. The general statement is the **backward pass of a broadcast**: *the gradient of a broadcast input is the sum of the incoming gradient over every axis that was stretched.* Three cases cover what a network uses:

- a vector $[O]$ stretched over the batch axis (the bias): sum over axis $0$, result $[O]$;
- a column $[B, 1]$ stretched over the output axis (a per-row scale or offset): sum over axis $1$, and *keep* the axis, so the result is again $[B, 1]$;
- a single number stretched over everything: sum over all entries.

<!-- worked: layer_gradients.broadcast_backward -->
The gradient arriving at the broadcast result is dY = G = [[1, -1, 2, 0], [0, 2, -1, 1], [-2, 1, 0, 3]] ([3, 4]).

| the small input | broadcast over | gradient (sum over the broadcast axes) | largest difference from central difference |
|---|---|---|---:|
| bias b [4], added to every row | the batch axis (axis 0) | [-1, 2, 1, 4] | 1.1e-10 |
| column c [3, 1], added to every column | the output axis (axis 1) | [[2], [2], [2]] | 2.8e-10 |
| scalar s [], added to every entry | both axes | 6 | 7.7e-11 |
<!-- /worked -->

![The same cells in the forward and backward pass of a broadcast. Left: the bias vector of four values is copied down into all three rows of a three by four array. Right: a three by four gradient dY, with entries 1, -1, 2, 0 in the first row, is summed down each column to give the bias gradient -1, 2, 1, 4.](img/broadcast-backward.svg)

The figure shows the mirror: in the forward pass one cell is copied into three rows, and in the backward pass three cells are added into one. The bias row of the numbers above is the one we met in the problem: the column-1 gradients $-1$, $2$, $1$ sum to $2$.

The same index map describes both directions. The forward pass writes $Y_{bo} = X_{bo} + b_{(0, o)}$ where the bias is read at a row index that is always $0$ (the stretched axis maps every index to $0$). The backward pass uses the same map to decide where each contribution is added:

```text
function unbroadcast(dY: array<real>[B, O], r: int, c: int) -> array<real>[r, c]:
    # the input before broadcasting had shape [r, c], with r in {1, B} and c in {1, O};
    # a bias [O] is [1, O] once a leading axis of size 1 is added
    require r == 1 or r == B, "the row axis is either stretched (1) or the batch size"
    require c == 1 or c == O, "the column axis is either stretched (1) or the output size"
    out = zeros(real, [r, c])
    for i in 0 .. B:
        for j in 0 .. O:
            ii = 0 if r == 1 else i            # a stretched axis maps every index to 0
            jj = 0 if c == 1 else j
            out[ii, jj] = out[ii, jj] + dY[i, j]     # the forward read becomes a backward add
    return out
```

An operation with one broadcast input then has a uniform backward: the gradient of the *large* input is `dY` itself (unchanged, for an addition), and the gradient of the *small* one is `unbroadcast(dY, ...)`. For a multiplication, `dY` is first multiplied by the other factor, as in the table of binary operations, and then summed down. In all cases the dimensions that the stretched axis had are gone from the gradient, summed away, and the result has the shape of the input.

## Moving data: reshape, transpose, slice, gather

Some operations compute nothing. They only choose which input value appears at which output position, $y_i = x_{\pi(i)}$ for an index map $\pi$. The backward pass is the bookkeeping of our one rule: each forward read of $x_{\pi(i)}$ by output $i$ becomes an addition of $dy_i$ into $dx_{\pi(i)}$:

$$
dx_p = \sum_{i\,:\,\pi(i) = p} dy_i .
$$

What happens depends on how many outputs read each input:

- **Reshape and transpose** read every input exactly once (the map $\pi$ is a permutation of positions). The backward pass moves each gradient entry back to where it came from: the inverse reshape, the inverse transpose. No arithmetic.
- **Slice** reads a part of the input, once each, and never reads the rest. The gradient of the part is $d\mathbf{y}$ and the gradient of everything else is zero: the backward builds a zero array of the input's shape and writes $d\mathbf{y}$ into the sliced region. A concatenation is the opposite: its backward slices the gradient apart.
- **Gather** (looking up rows by an index list) may read the same input row more than once if an index repeats. Those rows then receive the *sum* of the gradients of all their uses. This is the same accumulation as a shared weight, and it is the gradient of an embedding table, taken up in its own chapter.

<!-- worked: layer_gradients.index_ops -->
Input X = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]] ([3, 4]). Upstream gradients are drawn as the numbers 1, 2, 3, ... in row-major order of the result, so every entry of the answer can be traced.

| operation | result shape | upstream dY | gradient dX | largest difference from central difference |
|---|---|---|---|---:|
| reshape to [2, 6] | [2, 6] | [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]] | [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]] | 9.0e-09 |
| transpose | [4, 3] | [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]] | [[1, 4, 7, 10], [2, 5, 8, 11], [3, 6, 9, 12]] | 9.0e-09 |
| slice X[1:3, 1:3] | [2, 2] | [[1, 2], [3, 4]] | [[0, 0, 0, 0], [0, 1, 2, 0], [0, 3, 4, 0]] | 3.0e-09 |
| gather rows [2, 0, 2] | [3, 4] | [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]] | [[5, 6, 7, 8], [0, 0, 0, 0], [10, 12, 14, 16]] | 1.2e-08 |
<!-- /worked -->

In the gather row, rows $[2, 0, 2]$ were taken from the input. Input row $2$ was read twice, so its gradient is the sum of the first and third upstream rows: $(1+9, 2+10, 3+11, 4+12) = (10, 12, 14, 16)$. Input row $1$ was never read and gets zeros. Any implementation that assigns where this one adds would return $(9, 10, 11, 12)$ for row $2$: the last use wins and the first is silently lost.

## Gradient accumulation

Everything above produced the gradient of an operation's inputs *given* the gradient of its output. One more rule connects those pieces into a training step, and it is the one that produces the quietest bugs. When a value is used in several places, its gradient is the sum of the contributions of every place. The implementation of "sum of contributions" in a program that handles the places one at a time is **gradient accumulation**: each contribution is *added into* the gradient buffer, never assigned into it.

### A weight used twice

A weight matrix applied to two inputs, $\mathbf{z}_a = \mathbf{W}\mathbf{x}_a$ and $\mathbf{z}_b = \mathbf{W}\mathbf{x}_b$, is read by both. Its gradient is $d\mathbf{W} = d\mathbf{z}_a\,\mathbf{x}_a^\top + d\mathbf{z}_b\,\mathbf{x}_b^\top$, the sum of one outer product per use (this is the batched shared-weight rule again, with a batch of two). A backward pass visits the two uses one after the other, because the graph is walked in reverse order. If the first visit writes the gradient buffer and the second *assigns* to it, the first contribution is destroyed. The loss still falls, because the surviving term is a real gradient of part of the objective. The result is wrong by exactly the missing term:

<!-- worked: layer_gradients.shared_weight -->
One weight matrix W = [[1, -1, 0.5], [0, 2, 1]] is applied to two inputs, x_a = [1, 2, -1] and x_b = [0.5, -1, 3], giving z_a = W x_a and z_b = W x_b. The loss is L = g_a . z_a + g_b . z_b with g_a = [1, -2], g_b = [3, 1].

| way of combining the two uses | dW [2, 3] | largest difference from central difference |
|---|---|---:|
| assign, the first use is kept (g_a x_a^T) | [[1, 2, -1], [-2, -4, 2]] | 9.00 |
| assign, the second use is kept (g_b x_b^T) | [[1.5, -3, 9], [0.5, -1, 3]] | 4.00 |
| accumulate: add both | [[2.5, -1, 8], [-1.5, -5, 5]] | 4.7e-10 |
<!-- /worked -->

Weights shared on purpose appear throughout later volumes: a convolution applied at every position, a recurrent cell applied at every time step, an embedding table reused as the output projection (tied embeddings). In every one the backward pass adds the contribution of each use into the same buffer, and the buffer must have been zeroed at the start of the step. The two requirements go together. *Accumulate* because there are several contributions; *zero once per step* because the buffer persists between steps (Chapter 4's "zero before you accumulate"). A buffer that is assigned loses contributions; a buffer that is added into but never zeroed adds this step's gradient to every earlier step's.

### Many small batches, one big one

The second use of the same rule is across calls. Chapter 4 showed that the memory held for the backward pass grows with the batch size. A batch that does not fit can be split into **micro-batches**: run forward and backward on each piece, adding each piece's gradient into the same buffers, and take the optimiser step once. The sum over the pieces is the gradient of the whole, as long as every piece's contribution is scaled the way the whole batch would have scaled it.

That condition is where the mistakes are. Suppose the loss is a mean over examples, and the batch of $B$ examples is split into $K$ micro-batches of sizes $B_1, \dots, B_K$. The gradient we want is

$$
\nabla L = \frac{1}{B}\sum_{n=1}^{B} \nabla \ell_n ,
$$

with $\ell_n$ the loss of example $n$. A micro-batch $j$ computes the gradient of *its own* mean, $\nabla L_j = \frac{1}{B_j}\sum_{n \in j}\nabla\ell_n$. The relation between the two is

$$
\nabla L = \sum_{j=1}^{K} \frac{B_j}{B}\,\nabla L_j .
$$

For equal sizes $B_j = B/K$ the weights are all $1/K$: accumulate the sum of the micro-batch gradients, then multiply by $1/K$. For unequal sizes the weights differ, and a flat $1/K$ is wrong. The least error-prone design avoids the issue: have each micro-batch compute the gradient of *its examples' sum divided by the total count* $B$, then a plain sum is the answer for any split.

<!-- worked: layer_gradients.microbatch_accumulation -->
Seed 10560831267439448640. Draw order, all from `normal()`: `X` (8 x 3, row-major), `T` (8 x 2), then `W` (2 x 3) each multiplied by 0.5; the bias `b` is zero. Loss: the mean over all B O = 16 squared errors; micro-batches are consecutive rows. The one-batch gradient has norm 1.385759 (dW only).

| strategy | largest abs. difference of dW from the one-batch gradient | norm of dW / norm of one-batch dW |
|---|---:|---:|
| 4 micro-batches of 2, own means, summed | 3.1e+00 | 4.0000 |
| 4 micro-batches of 2, own means, summed, times 1/4 | 2.2e-16 | 1.0000 |
| micro-batches of 3, 3, 2, own means, times 1/3 | 1.6e-01 | 1.0400 |
| micro-batches of 3, 3, 2, own means, times B_j / B | 2.2e-16 | 1.0000 |
| micro-batches of 3, 3, 2, divided by the total B O | 2.2e-16 | 1.0000 |
<!-- /worked -->

The first row is the mean-versus-sum pitfall: forgetting to rescale gives a gradient $K = 4$ times too large, with a norm ratio of exactly $4$. The third row is the subtle one: splitting $8$ examples as $3, 3, 2$ and scaling by the flat $1/3$ gives a gradient whose norm is $4\%$ too large and whose direction is off by an entry-wise error of $0.16$, which looks like noise and is a bias. The last two rows are exact to rounding, by weighting each piece by $B_j / B$ or by dividing by the total count once. Accumulation also costs no extra arithmetic: the floating-point operations of the $K$ pieces add up to those of the whole, and only the memory for saved activations drops, roughly by $K$.

```text
function accumulate_gradient(X: array<real>[B, D], T: array<real>[B, O],
                             W: array<real>[O, D], b: array<real>[O],
                             cuts: list<int>) -> (array<real>[O, D], array<real>[O]):
    # cuts = [0, c1, ..., B] delimits consecutive micro-batches of rows
    require cuts[0] == 0 and cuts[len(cuts) - 1] == B, "the micro-batches must cover the batch exactly"
    dW = zeros(real, [O, D])                          # zeroed once, before the first micro-batch
    db = zeros(real, [O])
    for j in 0 .. len(cuts) - 1:
        Xj = X[cuts[j]:cuts[j + 1], :]
        Tj = T[cuts[j]:cuts[j + 1], :]
        Bj = cuts[j + 1] - cuts[j]
        Yj = matmul(Xj, transpose(W))
        dYj = zeros(real, [Bj, O])
        for i in 0 .. Bj:
            for o in 0 .. O:
                dYj[i, o] = 2.0 * (Yj[i, o] + b[o] - Tj[i, o]) / (B * O)   # the TOTAL count B O
        G = matmul(transpose(dYj), Xj)
        s = sum(dYj, axis=0)
        for o in 0 .. O:
            db[o] = db[o] + s[o]
            for d in 0 .. D:
                dW[o, d] = dW[o, d] + G[o, d]         # accumulate across micro-batches
    return (dW, db)
```

## Every rule, and what it saves

Here is the whole chapter in one table. "Needs" is what the forward pass must keep for the backward pass to read.

| operation | forward | backward | the backward needs |
|---|---|---|---|
| matmul | $\mathbf{C} = \mathbf{A}\mathbf{B}$ | $d\mathbf{A} = d\mathbf{C}\,\mathbf{B}^\top$, $d\mathbf{B} = \mathbf{A}^\top d\mathbf{C}$ | $\mathbf{A}$ and $\mathbf{B}$ |
| batched matmul, shared $\mathbf{B}$ | $\mathbf{C}_i = \mathbf{A}_i\mathbf{B}$ | $d\mathbf{A}_i = d\mathbf{C}_i\mathbf{B}^\top$, $d\mathbf{B} = \sum_i \mathbf{A}_i^\top d\mathbf{C}_i$ | $\mathbf{A}$ and $\mathbf{B}$ |
| linear | $\mathbf{Y} = \mathbf{X}\mathbf{W}^\top + \mathbf{b}$ | $d\mathbf{X} = d\mathbf{Y}\mathbf{W}$, $d\mathbf{W} = d\mathbf{Y}^\top\mathbf{X}$, $d\mathbf{b} = \sum_b d\mathbf{Y}$ | $\mathbf{X}$ (and $\mathbf{W}$, a parameter) |
| elementwise $f$ | $y = f(x)$ | $dx = dy\,f'(x)$ | $x$, or $y$, or a mask |
| add, subtract | $z = x \pm y$ | $dx = dz$, $dy = \pm dz$ | nothing |
| multiply | $z = x\,y$ | $dx = dz\,y$, $dy = dz\,x$ | $x$ and $y$ |
| divide | $z = x / y$ | $dx = dz / y$, $dy = -dz\,x/y^2$ | $x$ and $y$ |
| sum | $r = \sum_i x_i$ | $dx_i = dr$ | the shape of $x$ |
| mean of $n$ | $r = \frac1n\sum_i x_i$ | $dx_i = dr/n$ | the shape of $x$ |
| max | $r = \max_i x_i$ | $dx_i = dr$ at the argmax, else $0$ | the argmax index |
| softmax | $y = \mathrm{softmax}(x)$ | $dx_i = y_i(dy_i - \sum_j dy_j y_j)$ | $y$ |
| broadcast | stretch an axis | sum $dy$ over the stretched axes | the shapes |
| reshape, transpose | same entries, new layout | the inverse layout of $dy$ | the shapes |
| slice, concat | take or join parts | place $dy$ in zeros; slice $dy$ apart | the ranges, the sizes |
| gather | $y = x[\text{idx}]$ | zeros, then add each $dy$ row at its index | the indices |

The "needs" column is the memory bill of a training step. For tensors of $64 \times 4096$ in single precision:

<!-- worked: layer_gradients.saved_per_op -->
Tensors of shape [B, W] with B = 64, W = 4096 (262,144 entries), float32.

| operation | what the backward reads | entries or bits | bytes |
|---|---|---:|---:|
| linear Y = X W^T + b | the input X [B, W] | 262,144 | 1,048,576 |
| tanh, sigmoid | the output y | 262,144 | 1,048,576 |
| GELU, SiLU | the input x | 262,144 | 1,048,576 |
| ReLU | one bit per entry | 262,144 bits | 32,768 |
| softmax along W | the output y | 262,144 | 1,048,576 |
| max along W | the argmax index per row (4 bytes) | 64 | 256 |
| gather of 64 rows | the 64 row indices (4 bytes each) | 64 | 256 |
| sum, mean | only the sizes | 0 | 0 |
| add of a bias, reshape, transpose, slice | only the sizes | 0 | 0 |
<!-- /worked -->

The sums, means, reshapes and slices save nothing but sizes; the maximum and the gather save one index per row; ReLU saves a thirty-second of what a tanh does; and an operation that keeps a full tensor keeps one mebibyte of it here, which Chapter 4's memory table multiplied by depth and by batch size.

Finally the check. For every rule of the table we drew random inputs from the book's generator, computed the rule's gradient and the central difference of $L = \sum G\,f(\text{inputs})$ with respect to every entry of every input, and report the largest difference:

<!-- worked: layer_gradients.rule_checks -->
Seed 5575565785777740154. One generator for the whole table, `normal()` draws in table order: for each row its inputs in the order listed (row-major), then the output gradient (row-major), which has the shape of the row's output (a single number for the sum of all entries, [4] for the sum along axis 0, [3] for the mean and the max along axis 1). For the division the denominator is replaced by 1.5 plus the absolute value of its draw. Central-difference step 1e-6; the closest any ReLU or max input comes to its kink is 0.051 (a check on your draw order). The last column is rounding noise of the central difference: expect values below 1e-8, not these digits. [x > 0] in the ReLU rule is the Iverson bracket (1 when the condition holds, 0 otherwise). Phi and phi in the GELU rule are the standard Gaussian cumulative distribution, Phi(x) = (1 + erf(x / sqrt 2)) / 2, and density, phi(x) = exp(-x^2 / 2) / sqrt(2 pi).

| operation | backward rule | inputs | largest absolute difference |
|---|---|---|---:|
| linear X W^T + b | dX = dY W; dW = dY^T X; db = sum over batch of dY | [3, 2], [4, 2], [4] | 3.6e-10 |
| matmul A B | dA = dC B^T; dB = A^T dC | [2, 3], [3, 4] | 9.2e-11 |
| batched matmul A[i] B[i] | the matmul rule for every i | [2, 2, 3], [2, 3, 4] | 6.3e-10 |
| batched matmul, shared B | dA[i] = dC[i] B^T; dB = sum over i of A[i]^T dC[i] | [2, 2, 3], [3, 4] | 3.6e-10 |
| tanh | dx = dy (1 - y^2) | [3, 4] | 1.9e-10 |
| sigmoid | dx = dy y (1 - y) | [3, 4] | 3.9e-11 |
| ReLU | dx = dy [x > 0] | [3, 4] | 1.5e-10 |
| GELU | dx = dy (Phi(x) + x phi(x)) | [3, 4] | 7.5e-11 |
| SiLU | dx = dy s (1 + x (1 - s)), s = sigmoid(x) | [3, 4] | 1.5e-10 |
| x + y | dx = dz; dy = dz | [2, 3], [2, 3] | 1.8e-10 |
| x - y | dx = dz; dy = -dz | [2, 3], [2, 3] | 1.4e-10 |
| x * y | dx = dz y; dy = dz x | [2, 3], [2, 3] | 3.8e-10 |
| x / y | dx = dz / y; dy = -dz x / y^2 | [2, 3], [2, 3] | 8.6e-11 |
| sum of all entries | dx = dr in every entry | [3, 4] | 1.2e-10 |
| sum along axis 0 | dx[i, j] = dr[j] | [3, 4] | 2.0e-10 |
| mean along axis 1 | dx[i, j] = dr[i] / n | [3, 4] | 6.3e-11 |
| max along axis 1 | dr[i] to the first argmax, 0 elsewhere | [3, 4] | 9.2e-11 |
| softmax along axis 1 | dx = y (dy - sum of dy y) | [3, 4] | 7.1e-11 |
| add bias [4] to [3, 4] | dY; db = sum over axis 0 of dY | [3, 4], [4] | 1.1e-10 |
| add column [3, 1] to [3, 4] | dY; dc = sum over axis 1 of dY, kept as [3, 1] | [3, 4], [3, 1] | 3.6e-10 |
| add scalar to [3, 4] | dY; ds = sum of all of dY | [3, 4], [] | 1.1e-10 |
| reshape [3, 4] to [2, 6] | reshape dY back | [3, 4] | 1.2e-10 |
| transpose | transpose dY | [3, 4] | 8.4e-11 |
| slice [1:3, 1:3] | dY placed in a zero tensor | [3, 4] | 1.4e-10 |
| concat along axis 1 | slice dY apart | [2, 3], [2, 2] | 4.7e-11 |
| gather rows [2, 0, 2] | add each dY row into its source row | [3, 4] | 1.8e-10 |

Largest difference over all rows: 6.3e-10.
<!-- /worked -->

All 26 rows agree to about nine digits, the accuracy of a central difference at step $10^{-6}$ in double precision. Rounding, not the rule, sets that floor; the next chapter treats step size, rounding and tolerances properly. For now the table is the evidence that the rules above are not just derived but also correct.

## What the backward pass costs

The three matrix products of a linear layer have something in common that is easy to miss. Count them with the sizes $[B, D]$, $[O, D]$ and $[B, O]$:

- forward $\mathbf{Y} = \mathbf{X}\mathbf{W}^\top$: for each of $B \cdot O$ outputs, $D$ multiplications and $D$ additions;
- backward $d\mathbf{X} = d\mathbf{Y}\mathbf{W}$: for each of $B \cdot D$ outputs, $O$ multiplications and $O$ additions;
- backward $d\mathbf{W} = d\mathbf{Y}^\top\mathbf{X}$: for each of $O \cdot D$ outputs, $B$ multiplications and $B$ additions.

All three are $2BDO$ operations. Each reads two of the three arrays $\mathbf{X}$, $\mathbf{W}$ and $\mathbf{Y}$ (or their gradients, which have the same shapes) and writes the third, so they also move the same number of bytes. The backward pass of a linear layer is two matrix products where the forward is one, at the same cost each:

<!-- worked: layer_gradients.work_counts -->
Sizes B = 64, D = 784, O = 256, float32 (4 bytes). A multiply-add counts as two FLOPs. Bytes are the minimum traffic: each operand read once, the result written once.

| product | reads | writes | FLOPs | bytes moved | FLOPs per byte |
|---|---|---|---:|---:|---:|
| forward  Y = X W^T | X [B, D], W [O, D] | Y [B, O] | 25,690,112 | 1,069,056 | 24.03 |
| backward dX = dY W | dY [B, O], W [O, D] | dX [B, D] | 25,690,112 | 1,069,056 | 24.03 |
| backward dW = dY^T X | dY [B, O], X [B, D] | dW [O, D] | 25,690,112 | 1,069,056 | 24.03 |

| smaller operation | FLOPs | note |
|---|---:|---|
| bias add (forward) | 16,384 | one add per output entry |
| bias gradient | 16,384 | one add per entry of dY |

A stack of identical layers, where the first layer needs no dX because the data need no gradient:

| layers in the stack | forward matmul FLOPs | backward matmul FLOPs | backward / forward |
|---:|---:|---:|---:|
| 1 | 25,690,112 | 25,690,112 | 1.0000 |
| 2 | 51,380,224 | 77,070,336 | 1.5000 |
| 4 | 102,760,448 | 179,830,784 | 1.7500 |
| 8 | 205,520,896 | 385,351,680 | 1.8750 |
| 32 | 822,083,584 | 1,618,477,056 | 1.9688 |
<!-- /worked -->

**The backward pass of a matrix multiply costs about twice its forward pass** in arithmetic: $4BDO$ against $2BDO$ when the layer needs both $d\mathbf{X}$ and $d\mathbf{W}$. The first layer of a network needs no $d\mathbf{X}$, because the data are not a parameter and nothing below them needs a gradient; that layer's backward costs the same as its forward. A stack of $L$ identical layers therefore costs $(2L - 1)/L$ forwards in its backward, which tends to $2$ as depth grows; the table shows $1.9688$ at $32$ layers. A training step is a forward plus a backward, about three forwards of arithmetic, the figure Chapter 4 used. Everything else in the table is cheap by comparison: the bias add and the bias gradient are $BO$ operations each, against $2BDO$, a fraction $1/(2D)$ of one product. They are limited by moving data and not by arithmetic, which is why engines spend effort on them (the next section has an example).

## Making it work

**Check the shape of every gradient.** The gradient of an array has that array's shape. A $d\mathbf{W}$ of the wrong shape is a transposed product; one that has the right shape only because $B = O$ is the bug that distinct test sizes exist to expose.

**Add, and zero once.** Every backward writer of a parameter gradient adds. The buffer is cleared at the start of the step, on every path that will write it, including paths whose results seem unused.

**Never alias a saved buffer.** A backward rule reads a forward value. If a later forward stage overwrites the buffer (to save memory), the rule reads the wrong numbers and nothing complains. Chapter 4's story about aliased activation buffers is this failure.

**Do not forget the $1/n$.** A mean's backward divides by the count. A loss that sums where it should average is a gradient scaled by the count, and an adaptive optimiser may hide it for a long time.

**Test rules away from kinks.** ReLU at $0$, a tied maximum, and the absolute value at $0$ have no derivative; a central difference there reports the average of the one-sided slopes, which no rule reproduces. Pick test inputs at a distance from them larger than the finite-difference step.

## Where this appears in brain

brain's backward passes are written by hand as one kernel per rule (Chapter 4 explained why), so the rules of this chapter are visible as files in `crates/kernels/wgsl/`.

**Matrix products.** `matmul_dx.wgsl` computes $dX[m, k] = \sum_n dY[m, n]\,W[n, k]$ with one thread per output element and a serial sum over $n$. It takes an `accumulate` flag: $0$ overwrites, $1$ adds, "so dX can collect contributions from several matmuls", for example a shared input read by several experts of a mixture. `matmul_dw.wgsl` is $dW[n, k] \mathrel{+}= \sum_m dY[m, n]\,X[m, k]$ and always adds (`dw[idx] = dw[idx] + acc`); its header states that the weight-gradient buffer "is zeroed once before the backward pass" and that adding is what lets a tied embedding collect the contributions of both the output projection and the lookup. The two kernels are the two products of the section on matrix multiplication, and the two loops are the same length, $2BDO$ operations each, as the work table showed.

**Bias gradients and broadcasts.** `bias_grad.wgsl` is $dbias[n] \mathrel{+}= \sum_m dy[m, n]$, the sum over the broadcast axis, adding into a buffer zeroed beforehand. Its story is about the *shape* of the reduction. One thread per output feature walks all $m$ rows serially; when $m$ is the batch times the spatial extent of a convolution, tens of thousands of rows, only $n$ threads are launched against a device with far more lanes. Commit `eb3172799` (2 September 2026) replaced it for the convolutional models with a two-stage form: `bias_grad_part.wgsl` computes partial column sums over chunks of rows in parallel, and `bias_grad_final.wgsl` folds the partials and accumulates (`dbias[col] += sum of the partials`). The arithmetic is unchanged ($m \cdot n$ additions); what changed is how many lanes share it. The commit's own check was the finite-difference checker of every bias gradient in a real backward pass, the preview of the next chapter. `add_chan_bcast_dv.wgsl`, the gradient of adding a per-image-per-channel scalar to a feature map, says it in its header: "the adjoint of a broadcast is a sum over the broadcast axes"; the gradient with respect to the map itself "is dy itself", so the caller reuses the buffer and no kernel runs.

**Elementwise backward kernels.** Each activation has a `_bwd` kernel that reads the *input* of the activation: `tanh_act_bwd.wgsl` ($dx = dy\,(1 - \tanh^2 x)$), `sigmoid_bwd.wgsl`, `gelu_bwd.wgsl`, `silu_bwd.wgsl`, `leaky_relu_bwd.wgsl`. The header of `sigmoid_bwd.wgsl` gives the reason: it takes the pre-activation "matching the convention of every other `*_bwd` kernel here", so that blocks can cache one activation per stage. That is the other choice from the table in this chapter (saving the output is cheaper for tanh and sigmoid, saving the input works for all of them), taken for uniformity. `leaky_relu_bwd.wgsl` passes the gradient where $x \geq 0$, so at exactly zero it takes the slope-one branch, a different convention from the one we chose for ReLU; the engine picks one per kernel, and its gradient check keeps clear of the kink.

**Reductions that route.** The backward of a max pool, `maxpool2d_dx.wgsl`, is the argmax rule: the forward stores the winning index of every window, and the backward sends each window's gradient to it. Its header states the tie rule, the "frozen-argmax convention": the forward records *one* winner, "so a plateau sends all of dy to that single input". It is written in gather form, one thread per input element summing the gradients of the windows that chose it, so every `dx` entry is written by exactly one thread, "a plain overwrite", and needs no zeroing; an input read by several overlapping windows still receives the sum, which is the accumulate rule realised inside one thread. The softmax backward `softmax_k_dx.wgsl` is the boxed formula of the reductions section as two loops, one for the dot product and one for $y_k(dy_k - \mathrm{dot})$, and takes the forward *output*, as the chapter said it should (Chapter 3 tells its history).

**Index kernels.** `emb_bwd.wgsl` is the gather backward: for each vocabulary row it loops the tokens and adds the gradient of every token equal to that row (`grad_emb[v, c] += sum of d_x[n, c] over tokens n with token n = v`), which is the repeated-index rule. One thread per (row, channel) loops over the tokens, which avoids races without atomic additions. `row_scatter.wgsl` is the opposite kind: it *assigns*, so rows it never names are left untouched and the caller must zero the buffer when zeros are required.

**Accumulation across micro-batches.** The training loop in `crates/model/src/train.rs` does what the section prescribes: zero the gradients (`model.zero_grads()`), run `grad_accum` micro-steps that each add into the same buffers, then take one optimiser step with the scale $1/\mathrm{grad\_accum}$, "average grads over accumulation steps". That is the equal-size case of the rule; it relies on every micro-batch being the same size. `ParamStore::zero_grads` in `crates/paramstore/src/lib.rs` is the single place that clears gradient buffers.

Three defects from brain's history are this chapter's rules broken.

*A gradient that was never cleared* (commit `98e60aef9`, 20 October 2026; knowledge #334). `ParamStore::zero_grads` cleared the trainable and frozen parameters' gradients but left out those of parameters whose optimiser state lives in host memory (`Role::Offload`). A full fine-tune that used the host-memory optimiser therefore trained, from the second step on, on the sum of its own gradient and every earlier step's. The test that exposed it hashed all gradients of concurrent runs of one seed; at step 1 five of six runs agreed and one did not, and the odd run's gradient differed from the others' by exactly the step-0 gradient. The fix added the offload set to the clear (`zero_grads` now chains the three sets) and a regression test. The lesson is the rule: a buffer that is added into must be cleared on every path that writes it, and a loss assertion would not have found this, because the loss at the divergent step was bit-identical.

*Half a buffer assigned, half accumulated* (commit `002c3aeb8`, 25 September 2026; knowledge #157). A head wrote its hidden-state gradient into one buffer in two pieces: the state rows with an assigning kernel (`row_scatter`), the option rows with an accumulating one (`emb_bwd`). The assigned half corrected itself every step; the accumulated half carried the running sum of every step before it, because one backward path, the one for a frozen encoder, never cleared the buffer. Nothing in the crate read that half, so no test noticed; a regression test repeats one identical step twice and asserts the two gradients are equal (before the fix the second was exactly twice the first). The general lesson is in the section above: with a mix of assigning and accumulating writers into one buffer, the assigning one *masks* the missing clear of the other.

*The averaging scale* (commit `fab825f32`, 30 September 2026; knowledge #181). The optimiser entry point took an `extra_scale` that is the $1/K$ average of $K$ accumulated micro-batches, and the optimisers disagreed on what it meant: the device kernels multiplied by it, several host optimisers divided by it (a factor $K^2$ apart), and the clipping norm was measured on the unscaled sum. The test that compared two optimisers passed anyway, because Adam divides by the root of its second moment, which cancels a constant gradient scale unless $\varepsilon$ is comparable to it. The commit made one function, `optim::grad_multiplier`, the single definition, and tested it with a large $\varepsilon$ where scale shows. That is the "mean-versus-sum scaling pitfall" of the accumulation section, hidden for a time by the optimiser that Chapter 7 will introduce.

A fourth entry is the shared weight. In the vision tower of one imported model the same two convolutions run twice, once on the main path and once on a high-resolution branch whose output is added with a learned scale. The weight gradient is the sum of both uses, and the kernel `conv2d_dw.wgsl` accumulates into the parameter's buffer, so the sum came for free (commit `8d2b4f884`, 30 September 2026). The lesson is in the test, knowledge #187: a gradient check that projects the gradient on a few random directions passed even when one of the two uses was deliberately dropped, on one of the two weight tensors. A weight read at several sites is checked entry by entry (`crates/gradcheck/src/lib.rs`, `elementwise_check`; the gate is `check_sam_hd_shared` in `crates/gradcheck/src/sam_hd.rs`), at a coupling that is large enough for the second use to show.

## Check your understanding

Answers are at the end of the chapter.

1. **Find the error.** "For $\mathbf{Y} = \mathbf{X}\mathbf{W}^\top$ with $\mathbf{X}$ of shape $[B, D]$ and $\mathbf{W}$ of shape $[O, D]$, the weight gradient is $d\mathbf{W} = d\mathbf{Y}\,\mathbf{X}$." Check the shapes and give the correct formula.
2. **Predict.** A bias $\mathbf{b}$ of shape $[2]$ is added to every row of a $[3, 2]$ array, and the gradient arriving at the result is $[[1, 2], [3, 4], [5, 6]]$. What is $d\mathbf{b}$? What would it be if a column $\mathbf{c}$ of shape $[3, 1]$ had been added to every column instead?
3. **Explain.** Why does the backward of a mean divide by $n$, and what happens to the gradients of a mean loss over a batch if the division is forgotten?
4. **Predict.** For $y = \max(0, x)$ at $\mathbf{x} = (-1, 0, 2)$ with incoming $d\mathbf{y} = (3, 3, 3)$, what is $d\mathbf{x}$ under the convention $f'(0) = 0$, and what is the smallest thing the forward could have saved?
5. **Find the error.** "To save memory, store only the output of GELU and compute its backward from that." Give a concrete counterexample from the chapter.
6. **Predict.** $L = w x_1 + w x_2$ with $x_1 = 2$, $x_2 = 3$. What is $\frac{dL}{dw}$, and what would a backward pass that assigns instead of adding return?
7. **Explain.** A batch of $10$ examples is split into micro-batches of $4$, $3$ and $3$ and each micro-batch computes the gradient of its own mean loss. A colleague multiplies the sum of the three gradients by $1/3$. Is that the gradient of the mean over $10$? What weights are correct, and how does the "divide by the total count" design avoid the question?

## Exercises

1. **[Calculate]** A linear layer has $\mathbf{X} = \begin{pmatrix}1 & 2 & 0\\ 0 & 1 & 3\end{pmatrix}$ ($B = 2$, $D = 3$), $\mathbf{W} = (1, -1, 2)$ ($O = 1$), $\mathbf{b} = (0.5)$ and the gradient $d\mathbf{Y} = (2, -1)^\top$ arrives. Compute $d\mathbf{X}$, $d\mathbf{W}$ and $d\mathbf{b}$ and check the three shapes.

2. **[Calculate]** Derive the backward pass of centring a row, $\mathbf{y} = \mathbf{x} - \mathrm{mean}(\mathbf{x})$ (a subtraction of a broadcast scalar), using the mean, broadcast and subtraction rules in turn. Show that $d\mathbf{x} = d\mathbf{y} - \mathrm{mean}(d\mathbf{y})$, and that the entries of $d\mathbf{x}$ sum to zero.

3. **[Calculate]** For a batched product $\mathbf{A}_i\mathbf{B}$ with $N = 8$, $m = 16$, $k = 32$ and $n = 64$ and a shared $\mathbf{B}$, write the shapes of $d\mathbf{A}$ and $d\mathbf{B}$. Count the forward and backward multiply-adds, and the number of bytes of $d\mathbf{B}$ in single precision. Which sum in the backward has $N m$ terms?

4. **[Calculate]** For $\mathbf{x} = (2, 5, 5, 1)$ and an incoming $dr = 1$ for $r = \max(\mathbf{x})$, give the gradient under the first-index rule, under a rule that splits evenly among ties, and what a central difference reports at that point. Which of the three could a test compare an implementation against?

5. **[Calculate]** A batch of $10$ examples is split $4, 3, 3$ with each piece's gradient the gradient of its own mean. Give the weight each piece's gradient needs, and verify with a one-parameter model $L_n = (w x_n)^2$ that the weighted sum equals the gradient of the mean over all ten (choose your own $x_n$).

6. **[Implement]** Implement the batched linear layer's forward and `linear_backward` in your language. Reproduce the tables of `layer_gradients.linear_forward` and `layer_gradients.linear_backward`, then add a central-difference check over every entry of $\mathbf{X}$, $\mathbf{W}$ and $\mathbf{b}$ and reproduce `layer_gradients.linear_check` (the differences to the printed digits may vary with your floating-point library; the order of magnitude must match).

7. **[Implement]** Implement `unbroadcast` for the three cases (bias, column, scalar) and reproduce `layer_gradients.broadcast_backward`. Then write the backward of $\mathbf{Y} \odot \mathbf{s}$ with a column $\mathbf{s}$ of shape $[3, 1]$: what is the gradient of $\mathbf{s}$, and where is the $\odot$ factor applied relative to the sum?

8. **[Implement]** Implement the softmax backward and reproduce `layer_gradients.softmax_backward`. Check that rows of $d\mathbf{x}$ sum to zero, and that adding $100$ to every logit changes neither $\mathbf{y}$ nor $d\mathbf{x}$ when softmax subtracts the row maximum.

9. **[Implement]** Implement the SplitMix64 generator of Chapter 17, draw the data of `layer_gradients.microbatch_accumulation` in the stated order from the stated seed, and reproduce the five strategies of its table, including the factor $4$ and the $4\%$ error.

10. **[Implement]** Implement the batched shared-weight product, the gather and the shared-weight example, and reproduce `layer_gradients.batched_matmul`, `layer_gradients.index_ops` and `layer_gradients.shared_weight`. In the last, run the "assign" variant and confirm that its error equals the missing term exactly.

11. **[Investigate]** The rule table's last column, "the backward needs", decides what stays in memory. For a stack of $L$ layers of linear, GELU, linear, GELU and so on, with activations of $[B, W]$, list what each layer saves and compute the total in terms of $B$, $W$ and $L$. By how much does it drop if the GELU is replaced by a ReLU and only the mask is saved? (Reproduce the per-tensor bytes of `layer_gradients.saved_per_op` for your own sizes first.)

12. **[Investigate]** Run the central difference on $\max(\mathbf{x})$ and on ReLU at points $x = \delta$ for $\delta = 10^{-1}, 10^{-3}, 10^{-5}, 10^{-7}, 0$ with step $10^{-6}$. At which $\delta$ does the central difference stop agreeing with the rule? Explain it with the one-sided slopes, and say what a test of a network containing ReLUs should do about it.

## Summary

One formula, $dx_i = \sum_j dy_j\,\partial y_j/\partial x_i$, with one reading rule (every forward read becomes a backward add) generates the backward pass of every layer. The **backward pass of a linear layer** is $d\mathbf{X} = d\mathbf{Y}\mathbf{W}$, $d\mathbf{W} = d\mathbf{Y}^\top\mathbf{X}$ and $d\mathbf{b} = \sum_b d\mathbf{Y}$, which the shapes rebuild if forgotten. The **backward pass of a matrix multiply** is $d\mathbf{A} = d\mathbf{C}\mathbf{B}^\top$ and $d\mathbf{B} = \mathbf{A}^\top d\mathbf{C}$, per item for a batch, with a sum over the batch for a shared operand. The **backward pass of an elementwise function** is $dx = dy\,f'(x)$, and which of input, output or a mask must be saved depends on the function (GELU needs its input). The **backward pass of a reduction** copies the gradient to everything the reduction collapsed (sum), divides by the count (mean), routes it to the argmax with a stated tie rule (max), or applies the one-dot-product VJP of the softmax. The **backward pass of a broadcast** is the sum over every stretched axis, the mirror of the reduction rule, so copies forward are sums backward and sums forward are copies backward. Moving data (reshape, transpose, slice, gather) is index bookkeeping: gradients return to their sources, unread positions get zero, repeated reads add. **Gradient accumulation** is the principle behind all of them: contributions are added into a gradient buffer, never assigned, the buffer is zeroed once per step, and the sum of micro-batch gradients equals the whole batch's when each piece carries its share of the total count. Every rule has a central-difference check, and the backward of a matrix product costs two products of the forward's size, so a training step is about three forwards of arithmetic. In brain, the rules are one kernel each, the accumulate-or-assign choice of each kernel is stated in its header, and three of the knowledge entries are these rules broken: an uncleared buffer, half a buffer assigned and half accumulated, and a scale that only an optimiser's insensitivity hid.

We have taken each rule on trust from its derivation plus a quick numerical comparison. The next chapter, "Proving Gradients Correct", makes that comparison a proper instrument: how to choose the step, how rounding error limits it, how to judge a relative error, why a check must avoid kinks and square sizes, and how brain gates every model on it.

## Key terms

- **backward pass of a linear layer**: the three rules $d\mathbf{X} = d\mathbf{Y}\mathbf{W}$, $d\mathbf{W} = d\mathbf{Y}^\top\mathbf{X}$ and $d\mathbf{b} = \sum_b d\mathbf{Y}$ for $\mathbf{Y} = \mathbf{X}\mathbf{W}^\top + \mathbf{b}$, each gradient having the shape of the array it belongs to. <!-- concept: linear-layer-backward -->
- **backward pass of a matrix multiply**: for $\mathbf{C} = \mathbf{A}\mathbf{B}$, the gradients $d\mathbf{A} = d\mathbf{C}\mathbf{B}^\top$ and $d\mathbf{B} = \mathbf{A}^\top d\mathbf{C}$, two matrix products of the forward's size; for a batch, the same per item, summed over the batch for an operand shared by all items. <!-- concept: matmul-backward -->
- **backward pass of an elementwise function**: multiplying the incoming gradient entry by entry with the derivative of the function at the input, $dx = dy\,f'(x)$, which needs the input, the output or a mask saved from the forward. <!-- concept: elementwise-backward -->
- **backward pass of a reduction**: the rule that sends the gradient of a sum, mean or maximum back to the entries that were collapsed: copied for a sum, divided by the count for a mean, routed to the argmax for a maximum. <!-- concept: reduction-backward -->
- **backward pass of a broadcast**: the rule that sums the incoming gradient over every axis the forward pass stretched, so a copy in the forward becomes a sum in the backward. <!-- concept: broadcast-backward -->
- **gradient accumulation**: adding each contribution to a value's gradient into one persistent buffer, never assigning, with the buffer zeroed once per optimiser step; it sums the uses of a shared weight and the pieces of a batch split into micro-batches. <!-- concept: gradient-accumulation -->

## Answers

**Before you begin.**

1. $\frac{dz}{dx} = \sum_{k=1}^{3}\frac{\partial z}{\partial y_k}\frac{dy_k}{dx}$, one product per path from $x$ to $z$, added.
2. The Jacobian is $n \times n$ and diagonal, because $a_i$ depends only on $z_i$. The product of a row vector with a diagonal matrix scales entry $i$ by the diagonal entry $i$: $n$ multiplications.
3. $d\mathbf{Y}$ is $[B, O]$, $d\mathbf{W}$ is $[O, D]$ and $d\mathbf{X}$ is $[B, D]$: a gradient always has the shape of the array it is the gradient of.
4. Every row of the $[3, 4]$ array gets $\mathbf{b}$ added: axis $0$ is stretched (the missing leading axis of $\mathbf{b}$ is treated as size $1$ and repeated three times). No memory is copied; the same four numbers are read at a stride of zero along the stretched axis.
5. $(\mathbf{A}\mathbf{B})^\top = \mathbf{B}^\top\mathbf{A}^\top$.

**Check your understanding.**

1. $d\mathbf{Y}$ is $[B, O]$ and $\mathbf{X}$ is $[B, D]$: their product is undefined unless $O = B$, and then it has the wrong shape $[B, D]$ instead of $[O, D]$. The index $B$ is shared and must be summed over, so $d\mathbf{Y}$ is transposed first: $d\mathbf{W} = d\mathbf{Y}^\top\mathbf{X}$.
2. The bias is read by every row, so $d\mathbf{b}$ sums down the columns: $(1 + 3 + 5, 2 + 4 + 6) = (9, 12)$. A column $\mathbf{c}$ of shape $[3, 1]$ stretched over the output axis has its gradient summed along the rows and kept as $[3, 1]$: $(1 + 2, 3 + 4, 5 + 6)^\top = (3, 7, 11)^\top$.
3. Each input contributes $1/n$ of the mean, so its derivative is $1/n$ and $dx_i = dr/n$. Forgetting the division gives each entry a gradient $n$ times too large (for a mean over $n$ examples, the gradient of the sum loss), a scale that tests of loss decrease do not notice but that changes the effective learning rate by a factor $n$.
4. $d\mathbf{x} = (0, 0, 3)$: the entries at $-1$ and at $0$ get zero (the convention at the kink), the entry at $2$ passes the gradient. The forward needs to save only which entries were positive: one bit per entry.
5. GELU is not monotonic, so one output can come from two inputs with different derivatives. The output $-0.1$ comes from $x = -1.5017$ (derivative $-0.127$) and $x = -0.2489$ (derivative $+0.305$); a backward computed from the output alone cannot tell them apart. GELU saves its input.
6. $\frac{dL}{dw} = x_1 + x_2 = 5$, from two contributions $2$ and $3$. A backward that assigns would return $3$ or $2$ depending on which use it visits last, losing the other.
7. No. The mean over $10$ is $\frac{4}{10}\nabla L_1 + \frac{3}{10}\nabla L_2 + \frac{3}{10}\nabla L_3$; the sum times $1/3$ weights each piece by $0.333$ and so gives the first piece too little weight and the other two too much. The correct weights are $B_j/B$. If each micro-batch instead computes the gradient of the sum of its examples' losses divided by the total count $10$, a plain sum of the three is exact for any split.

## References

- Rumelhart, D. E., Hinton, G. E. and Williams, R. J., *Learning representations by back-propagating errors*, 1986.
- Griewank, A. and Walther, A., *Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation*, second edition, 2008.
- Baydin, A. G., Pearlmutter, B. A., Radul, A. A. and Siskind, J. M., *Automatic differentiation in machine learning: a survey*, 2018.
