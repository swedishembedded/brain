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
| $[a, b]$, $(a, b)$ | closed and open intervals |
| $\{1, \dots, n\}$ | the integers from 1 to $n$ |
| $x \in S$ | $x$ is an element of the set $S$ |
| $\lvert S \rvert$ | number of elements of a finite set $S$ |
| $\approx$ | approximately equal |
| $:=$ | is defined as |

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
| $\lVert \mathbf{x} \rVert$ | Euclidean norm (length) of $\mathbf{x}$ |
| $\mathrm{diag}(\mathbf{x})$ | the diagonal matrix with $\mathbf{x}$ on its diagonal |

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
| $H$ | number of attention heads |
| $V$ | vocabulary size |
| $C$ | channels (images, audio) or classes (classification) |
| $N$ | number of examples in a dataset, or a generic count |

## Functions and calculus

| Symbol | Meaning |
|---|---|
| $f: A \to B$ | a function from set $A$ to set $B$ |
| $f \circ g$ | composition: $(f \circ g)(x) = f(g(x))$ |
| $\exp(x)$, $e^x$ | exponential function |
| $\log x$ | natural logarithm (base $e$) unless a base is written |
| $\frac{df}{dx}$, $f'(x)$ | derivative of $f$ |
| $\frac{\partial L}{\partial w}$ | partial derivative of $L$ with respect to $w$ |
| $\nabla_{\mathbf{w}} L$ | gradient of $L$ with respect to $\mathbf{w}$, same shape as $\mathbf{w}$ |
| $\mathbf{J}_f$ | Jacobian matrix of $f$ |
| $\sum_{i=1}^{n}$, $\prod_{i=1}^{n}$ | sum and product over $i = 1, \dots, n$ |
| $\arg\min_x f(x)$ | the $x$ at which $f$ is smallest |

## Probability and statistics

| Symbol | Meaning |
|---|---|
| $p(x)$ | probability (or density) of $x$ |
| $p(y \mid x)$ | probability of $y$ given $x$ |
| $x \sim p$ | $x$ is drawn from the distribution $p$ |
| $\mathbb{E}[X]$, $\mathbb{E}_{x \sim p}[f(x)]$ | expectation |
| $\mathrm{Var}[X]$ | variance |
| $\mathcal{N}(\mu, \sigma^2)$ | Gaussian with mean $\mu$, variance $\sigma^2$ |
| $H(p)$ | entropy |
| $H(p, q)$ | cross-entropy of $q$ relative to $p$ |
| $D_{\mathrm{KL}}(p \,\Vert\, q)$ | Kullback-Leibler divergence |

## Learning

| Symbol | Meaning |
|---|---|
| $\theta$ | all parameters of a model, as one long vector |
| $f_\theta$ | a model with parameters $\theta$ |
| $\mathcal{D} = \{(\mathbf{x}_n, y_n)\}_{n=1}^N$ | a dataset of $N$ examples |
| $\hat{y}$ | a prediction (the hat marks an estimate) |
| $L$, $\mathcal{L}$ | loss for one example, loss averaged over a batch or dataset |
| $\eta$ | learning rate |
| $t$ | training step, or time in a sequence or a diffusion process |
| $\sigma(x)$ | the logistic sigmoid $1/(1+e^{-x})$ |
| $\mathrm{softmax}(\mathbf{z})_i$ | $e^{z_i} / \sum_j e^{z_j}$ |

## Units

Sizes of memory are in bytes with binary prefixes: 1 KiB = 1024 bytes,
1 MiB = 1024 KiB, 1 GiB = 1024 MiB. Counts of parameters use decimal
prefixes: a 7B model has about $7 \times 10^9$ parameters. Rates of
arithmetic are in floating-point operations per second (FLOP/s), counting
a multiply and an add as two operations.
