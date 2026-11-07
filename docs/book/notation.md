<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Notation

This page collects the symbols used throughout the book. Each one is defined
properly in the chapter that introduces it; this is a reference, not a place
to learn them. Where the literature uses several conventions, the book picks
one and keeps to it.

## Numbers and sets

| Symbol | Meaning |
|---|---|
| $\mathbb{N}$, $\mathbb{Z}$, $\mathbb{R}$ | natural numbers $\{0, 1, 2, \dots\}$, integers, real numbers |
| $\mathbb{R}^n$ | vectors of $n$ real numbers |
| $\mathbb{R}^{m \times n}$ | matrices with $m$ rows and $n$ columns |
| $[a, b]$, $(a, b)$, $[a, b)$ | closed, open and half-open intervals; $(0, \infty)$ is all positive reals |
| $\{1, \dots, n\}$ | the integers from 1 to $n$ |
| $x \in S$, $x \notin S$ | $x$ is, is not, an element of the set $S$ |
| $\varnothing$ | the empty set |
| $A \subseteq B$ | every element of $A$ is in $B$ |
| $A \cup B$, $A \cap B$ | union (in $A$ or $B$) and intersection (in both) |
| $B \setminus A$, $A^c$ | difference (in $B$ but not in $A$) and complement of $A$ (everything not in $A$) |
| $A \times B$ | Cartesian product: all pairs $(a, b)$ with $a \in A$, $b \in B$ |
| $\lvert S \rvert$ | number of elements of a finite set $S$ |
| $\lvert x \rvert$ | absolute value of a number $x$ (modulus of a complex number) |
| $\lceil x \rceil$, $\lfloor x \rfloor$ | $x$ rounded up, and down, to an integer |
| $n!$ | factorial: $n(n-1)\cdots 1$, with $0! = 1$ |
| $\binom{n}{k}$ | binomial coefficient $n!/(k!\,(n-k)!)$: the number of ways to choose $k$ of $n$ things |
| $a \bmod n$ | remainder of integer division (the pseudocode's `mod`) |
| $\infty$ | infinity |
| $\iff$, $\Rightarrow$ | if and only if; implies |
| $\approx$ | approximately equal |
| $:=$ | is defined as |
| $\mapsto$ | "maps to": $x \mapsto x^2$ is the function that squares |

## Scalars, vectors, matrices and tensors

| Symbol | Meaning |
|---|---|
| $x$, $\alpha$ | a scalar (italic lowercase) |
| $\mathbf{x}$ | a column vector (bold lowercase) |
| $x_i$ | the $i$-th element of $\mathbf{x}$ |
| $\mathbf{W}$ | a matrix (bold uppercase) |
| $W_{ij}$ | the element in row $i$, column $j$ of $\mathbf{W}$ |
| $\mathbf{W}^\top$ | the transpose of $\mathbf{W}$ |
| $\mathbf{I}$ | the identity matrix |
| $\mathsf{X}$ | a tensor with more than two axes (sans-serif) |
| $\mathbf{x} \cdot \mathbf{y}$, $\mathbf{x}^\top \mathbf{y}$ | dot product |
| $\mathbf{x} \odot \mathbf{y}$ | elementwise (Hadamard) product |
| $\lVert \mathbf{x} \rVert$, $\lVert \mathbf{x} \rVert_2$ | Euclidean norm (length) of $\mathbf{x}$ |
| $\lVert \mathbf{x} \rVert_1$, $\lVert \mathbf{x} \rVert_\infty$ | sum of absolute values; largest absolute value |
| $\lVert \mathbf{A} \rVert_F$ | Frobenius norm of a matrix: square root of the sum of the squares of all its entries |
| $\hat{\mathbf{u}}$ | the unit vector $\mathbf{u}/\lVert\mathbf{u}\rVert$ in the direction of $\mathbf{u}$ (on a scalar or parameter the hat marks an estimate, see below) |
| $\mathbf{0}$, $\mathbf{1}$ | the vector of all zeros, of all ones |
| $\mathbf{e}_j$ | the vector with a 1 in position $j$ and zeros elsewhere |
| $\mathrm{diag}(\mathbf{x})$ | the diagonal matrix with $\mathbf{x}$ on its diagonal |
| $\mathbf{A}^{-1}$ | the inverse of a square matrix $\mathbf{A}$ |
| $\det\mathbf{A}$, $\mathrm{tr}\,\mathbf{A}$ | determinant and trace (sum of the diagonal) |
| $\mathrm{rank}\,\mathbf{A}$ | the **rank of a matrix**: its number of independent columns (Chapter 4) |
| $\mathbf{a}_{i:}$, $\mathbf{a}_{:j}$ | row $i$ and column $j$ of $\mathbf{A}$ |
| $\mathbf{Q}$ | an orthogonal matrix: $\mathbf{Q}^\top\mathbf{Q} = \mathbf{I}$ (Chapter 6) |
| $\mathbf{R}(\theta)$ | the matrix that rotates the plane by the angle $\theta$ |
| $\lambda$, $\boldsymbol{\Lambda}$ | an eigenvalue; the diagonal matrix of eigenvalues |
| $\mathbf{U}$, $\boldsymbol{\Sigma}$, $\mathbf{V}$ | the factors of the singular value decomposition $\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$; $\sigma_i$ is the $i$-th singular value |
| $\mathbf{A}_r$ | the rank-$r$ truncation of $\mathbf{A}$ (keep the $r$ largest singular values) |
| $\omega$ | a rotation rate: angle per unit of position (Chapter 6), or an angular frequency (Chapter 14) |
| $\Delta$ | $ad - bc$, the determinant of a $2 \times 2$ matrix, in Chapter 4; with a following symbol, $\Delta t$, a small step |
| $d_{\mathrm{in}}$, $d_{\mathrm{out}}$ | input and output widths of a layer |
| $\mathrm{proj}_{\mathbf{u}}\mathbf{v}$, $\mathrm{cos\_sim}(\mathbf{a}, \mathbf{b})$ | the projection of $\mathbf{v}$ on $\mathbf{u}$; the cosine similarity of two vectors |
| $\mathrm{i}$ | the imaginary unit, $\mathrm{i}^2 = -1$ (the pseudocode writes `1i`) |
| $\mathrm{Re}\,z$, $\mathrm{Im}\,z$, $\bar z$ | real part, imaginary part and complex conjugate of $z$ |

A tensor has a **number of axes** (its order), written $R$ in Chapter 5; the
word *rank* is kept for matrices, as in the table above.

Mathematical indices start at 1. Array indices in pseudocode start at 0
(see [Pseudocode Conventions](pseudocode.md)). When the two meet, the text
says which is meant.

Shapes are written in brackets, outermost axis first:
$[B, T, D]$ is a tensor of $B$ sequences, each of $T$ positions, each a
vector of $D$ numbers. The letters below are used consistently for shapes.

| Letter | Usual meaning |
|---|---|
| $B$ | batch size: number of independent examples processed together |
| $T$ | sequence length: number of positions (tokens, frames, time steps) |
| $D$ | model width: length of the vector at each position |
| $H$ | number of attention heads, or the height of an image |
| $V$ | vocabulary size |
| $C$ | channels (images, audio) or classes (classification) |
| $N$ | number of examples in a dataset, or a generic count |

The letter $H$ has two more meanings, always distinguishable: bold
$\mathbf{H}$ is the Hessian matrix, and $H(p)$, with an argument, is entropy.

## Functions and calculus

| Symbol | Meaning |
|---|---|
| $f: A \to B$ | a function from set $A$ to set $B$ |
| $f \circ g$ | composition: $(f \circ g)(x) = f(g(x))$ |
| $f^{-1}$ | the inverse function of $f$ (not $1/f$) |
| $\exp(x)$, $e^x$ | exponential function |
| $\log x$, $\ln x$ | natural logarithm (base $e$) unless a base is written; $\log_b x$ is the logarithm to base $b$ |
| $\frac{df}{dx}$, $f'(x)$ | derivative of $f$ |
| $\frac{\partial L}{\partial w}$ | partial derivative of $L$ with respect to $w$ |
| $\nabla_{\mathbf{w}} L$ | gradient of $L$ with respect to $\mathbf{w}$, same shape as $\mathbf{w}$ |
| $\mathbf{J}_f$ | Jacobian matrix of $f$ |
| $\mathbf{H}$ | Hessian matrix of second partial derivatives |
| $D_{\mathbf{u}} f$ | directional derivative of $f$ along the unit vector $\mathbf{u}$ |
| $\Delta t$, $h$ | a small step in $t$ (or $x$): the step size of a numerical method |
| $\int_a^b f(x)\,dx$ | the integral of $f$ from $a$ to $b$: the area under the graph |
| $\sum_{i=1}^{n}$, $\prod_{i=1}^{n}$ | sum and product over $i = 1, \dots, n$ (the large upright $\sum$ is always a sum) |
| $\arg\min_x f(x)$, $\arg\max_x f(x)$ | the $x$ at which $f$ is smallest, largest (the first such $x$ on a tie) |
| $\max$, $\min$ | largest, smallest of the numbers listed |
| $\mathrm{sign}(x)$ | $+1$ for $x \ge 0$ and $-1$ for $x < 0$, so $\mathrm{sign}(0) = +1$ (the pseudocode page explains why) |
| $\mathrm{erf}(x)$ | the error function $\frac{2}{\sqrt\pi}\int_0^x e^{-s^2}ds$ |
| $\varepsilon$ | machine epsilon: the gap between 1 and the next larger floating-point number ($2^{-52}$ in 64 bits); $u = \varepsilon/2$ is the unit round-off |
| $\mathrm{fl}(x)$ | the floating-point number a computer stores for $x$ |
| $O(\cdot)$, $\Theta(\cdot)$, $\Omega(\cdot)$ | growth rates of a cost (Chapter 16) |
| $\oplus$, $\gg$ | bitwise exclusive-or, and shift right (the pseudocode's `xor`, `>>`) |
| $\gamma$ | a decay factor, e.g. in a learning-rate schedule (Chapter 2) |

## Probability and statistics

| Symbol | Meaning |
|---|---|
| $\Omega$ | the sample space: the set of all outcomes of an experiment |
| $P(A)$ | probability of an event $A \subseteq \Omega$ |
| $p(x)$ | probability (or density) of the value $x$ |
| $p(y \mid x)$ | probability of $y$ given $x$ |
| $\mathbf{1}_A$ | the indicator of $A$: 1 when the event occurs, 0 otherwise |
| $x \sim p$ | $x$ is drawn from the distribution $p$ |
| $\mathbb{E}[X]$, $\mathbb{E}_{x \sim p}[f(x)]$ | expectation |
| $\mathrm{Var}[X]$, $\mathrm{Cov}[X, Y]$ | variance and covariance |
| $\rho$ | correlation coefficient |
| $\boldsymbol{\Sigma}$ | a covariance matrix (Chapters 10 to 12); in Chapter 6 the diagonal matrix of singular values (see "Symbols with more than one meaning") |
| $\bar{x}$, $\bar{X}$ | the sample mean |
| $s$, $s^2$ | the sample standard deviation and variance (divisor $n - 1$ unless the text says $n$) |
| $\mathrm{SE}$ | standard error: the standard deviation of an estimate |
| $\hat{p}$, $\hat\theta$ | an estimate computed from data (the hat marks an estimate) |
| $\alpha$ | significance level of a test, or confidence level $1 - \alpha$ |
| $\mathcal{N}(\mu, \sigma^2)$ | Gaussian with mean $\mu$, variance $\sigma^2$ |
| $\phi(z)$, $\Phi(z)$ | density and cumulative distribution function of $\mathcal{N}(0, 1)$ |
| $\chi^2$ | the chi-square statistic or distribution |
| $H(p)$ | entropy |
| $H(p, q)$ | cross-entropy of $q$ relative to $p$ |
| $H(X \mid Y)$, $I(X; Y)$ | conditional entropy and mutual information |
| $D_{\mathrm{KL}}(p \,\Vert\, q)$ | Kullback-Leibler divergence |

## Learning

| Symbol | Meaning |
|---|---|
| $\theta$ | all parameters of a model, as one long vector |
| $f_\theta$ | a model with parameters $\theta$ |
| $\mathcal{D} = \{(\mathbf{x}_n, y_n)\}_{n=1}^N$ | a dataset of $N$ examples |
| $\hat{y}$ | a prediction (the hat marks an estimate) |
| $L$, $\mathcal{L}$ | loss for one example, loss averaged over a batch or dataset |
| $\eta$ | learning rate; $\eta_{\max}$, $\eta_{\min}$ are the largest and smallest rate of a schedule |
| $t$ | training step, or time in a sequence or a diffusion process |
| $\sigma(x)$ | the logistic sigmoid $1/(1+e^{-x})$ (always written with an argument) |
| $\mathrm{softmax}(\mathbf{z})_i$ | $e^{z_i} / \sum_j e^{z_j}$ |

## Symbols with more than one meaning

A few letters carry more than one meaning in the book. The text says which
is meant, and these rules decide the rest.

| Symbol | Meanings and how to tell them apart |
|---|---|
| $\sigma$ | $\sigma(x)$ with an argument in parentheses is the sigmoid. $\sigma_i$ with an index is the $i$-th singular value of a matrix (Chapter 6). Bare $\sigma$ and $\sigma^2$, as in $\mathcal{N}(\mu, \sigma^2)$, are a standard deviation and a variance |
| $\boldsymbol{\Sigma}$ and $\sum$ | bold $\boldsymbol{\Sigma}$ is a matrix: the diagonal matrix of singular values in Chapter 6, a covariance matrix in Chapters 10 to 12. The large upright $\sum$ is always the summation sign |
| $H$ | italic $H$ in a shape is the number of heads or the height of an image; bold $\mathbf{H}$ is the Hessian; $H(p)$ with an argument is entropy |
| $\Omega$ | alone, the sample space; followed by a function, $\Omega(n)$, a growth rate |
| rank | the rank of a matrix is its number of independent columns; a tensor has a number of axes, never a rank |
| hat | on a vector, $\hat{\mathbf{u}}$ is a unit vector; on a scalar or parameter ($\hat y$, $\hat p$) it is an estimate |

## Units

Sizes of memory are in bytes. Binary prefixes: 1 KiB = 1024 bytes,
1 MiB = 1024 KiB, 1 GiB = 1024 MiB. Decimal prefixes: 1 KB = $10^3$ bytes,
1 MB = $10^6$ bytes, 1 GB = $10^9$ bytes, so 1 GiB is about 7.4 percent
more than 1 GB. Counts of parameters use decimal
prefixes: a 7B model has about $7 \times 10^9$ parameters. Rates of
arithmetic are in floating-point operations per second (FLOP/s), counting
a multiply and an add as two operations.
