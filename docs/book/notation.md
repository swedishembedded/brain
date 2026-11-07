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
| $\pm$, $\mp$ | plus or minus; minus or plus (the lower sign goes with the lower sign of a $\pm$ in the same line) |
| $\propto$ | is proportional to: equal up to a constant factor that does not depend on the variable of interest |
| $a \equiv b \pmod m$ | congruent: $a - b$ is a multiple of $m$ (the same remainder on division by $m$) |
| $\gcd(a, b)$ | the greatest common divisor of two integers |
| $\ll$, $\lesssim$ | much less than; less than or about equal to, up to a modest constant (Chapters 6, 15) |
| $\leftarrow$ | assignment in a displayed update: $x \leftarrow x + 1$ replaces the value of $x$ |
| $\blacksquare$ | end of a proof or derivation |
| $a \div b$ | integer division, the pseudocode's `//` (Chapter 5) |
| $\Pi$ | the capital Greek letter pi: the same product as $\prod$, named as a word in Chapter 1 ("$\Pi$ means multiply"); the book writes $\prod$ |
| $x^\ast$ | a special point of $x$ in a derivation, such as the end of a segment or a fixed point (Chapters 7, 8); $x^\star$ in the calculus table is the minimiser |

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
| $\tilde{\mathbf{x}}$, $\tilde{\mathbf{W}}$ | in Chapter 4, the input with a 1 appended, $(\mathbf{x}, 1)$, and the weight matrix with the bias as an extra last column, $[\mathbf{W}\,|\,\mathbf{b}]$ |
| $\mathrm{op}(\mathbf{A})$, $\mathrm{ld}$ | in Chapter 5, $\mathbf{A}$ or its transpose according to a flag; the leading dimension, the stride between columns (or rows) of a stored matrix |
| $\operatorname{trace}\,\mathbf{A}$ | the same as $\mathrm{tr}\,\mathbf{A}$ (Chapter 6 writes it out) |
| $\overline{z}$, $\overline{\cdot}$ | the complex conjugate, as $\bar z$; a long bar over an expression, such as $\overline{\operatorname{DFT}(\overline{X})}$, conjugates every entry (Chapter 14) |
| $\omega$ | a rotation rate: angle per unit of position (Chapter 6), or an angular frequency (Chapter 14) |
| $\Delta$ | $ad - bc$, the determinant of a $2 \times 2$ matrix, in Chapter 4; with a following symbol, $\Delta t$, a small step |
| $d_{\mathrm{in}}$, $d_{\mathrm{out}}$ | input and output widths of a layer |
| $\mathrm{proj}_{\mathbf{u}}\mathbf{v}$, $\mathrm{cos\_sim}(\mathbf{a}, \mathbf{b})$ | the projection of $\mathbf{v}$ on $\mathbf{u}$; the cosine similarity of two vectors |
| $\mathrm{i}$ | the imaginary unit, $\mathrm{i}^2 = -1$ (the pseudocode writes `1i`) |
| $\mathrm{Re}\,z$, $\mathrm{Im}\,z$, $\bar z$ | real part, imaginary part and complex conjugate of $z$ |
| $\omega$, $X[k]$, $\mathbf{F}$ | in Chapter 14 only: $\omega = e^{-2\pi\mathrm{i}/N}$ is the primitive $N$-th root of unity, $X[k] = \sum_n x[n]\,\omega^{kn}$ the DFT of the signal $x$, and $\mathbf{F}$ the matrix with entries $\omega^{kn}$ (so $\mathbf{F}^H$ is its conjugate transpose, $\langle u, v\rangle$ the inner product conjugating its first argument) |

Words set upright inside a formula, such as $\mathrm{offset}$, $\mathrm{kv}$ or $\mathrm{head\_dim}$, are labels for quantities the text defines on the spot (the same words as in the pseudocode), not symbols with a fixed meaning.

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
| $\Gamma(z)$ | the gamma function, $\Gamma(n) = (n-1)!$ for a positive integer $n$ and $\Gamma(z+1) = z\Gamma(z)$ (used for the volume of a ball, Chapter 17, Exercise 5) |
| $\lim_{h \to 0}$ | the limit as $h$ approaches 0: the value the expression gets arbitrarily close to (Chapters 7, 8, 9) |
| $\tanh(x)$ | the hyperbolic tangent $(e^x - e^{-x})/(e^x + e^{-x})$ (Chapters 7, 8, 15) |
| $\mathrm{atan2}(y, x)$ | the angle of the point $(x, y)$ (the pseudocode's `atan2`; Chapter 3 and 14) |
| $\mathrm{relu}(x)$, $\mathrm{clamp}(x, l, u)$ | $\max(0, x)$ and $\min(\max(x, l), u)$, defined where they first appear (Chapter 2) |
| $\mathrm{log1p}(x)$ | $\log(1 + x)$ computed accurately for tiny $x$ (Chapter 15; the pseudocode's `log1p`) |
| $\epsilon$ | in Chapters 6, 11, 13 and 15, a small amount or a relative perturbation (as in $x(1 + \epsilon)$ or a probability shifted by $\epsilon$); $\varepsilon$ is reserved for machine epsilon and the tolerances listed in "Symbols with more than one meaning" |
| $\beta$ | in Chapters 6 and 14 only, a rotation angle (as $\alpha$ is); in Volume II the scale $\beta_w$ of a prior |
| $\mathrm{atol}$, $\mathrm{rtol}$ | absolute and relative tolerance of `close` (the pseudocode page) |
| $\varepsilon$, $\varepsilon_{\mathrm{mach}}$ | machine epsilon: the gap between 1 and the next larger floating-point number ($2^{1-p}$, which is $2^{-52}$ in 64 bits); Chapter 15 writes $\varepsilon_{\mathrm{mach}}$ |
| $u$ | the unit round-off $2^{-p} = \varepsilon_{\mathrm{mach}}/2$: the largest relative error of rounding one number (Chapter 15) |
| $\mathrm{fl}(x)$ | the floating-point number a computer stores for $x$ (the exact result rounded to the nearest, ties to even) |
| $\delta$, $\theta_i$ | in Chapter 15, the relative error of one rounding ($\lvert\delta\rvert \le u$) and the accumulated relative error of the $i$-th term of a sum |
| $\mathrm{lse}(z)$ | log-sum-exp: $\log \sum_i e^{z_i}$ (Chapter 15) |
| $s(x)$ | the logistic sigmoid, written $s$ in Chapter 7 (the same function as $\sigma(x)$) |
| $D_+(h)$, $D_0(h)$ | the forward and the central difference quotient with step $h$ (Chapter 7) |
| $x_n$, $e_n$ | the $n$-th iterate of an iteration and its error $x_n - x^\star$ (Chapter 7) |
| $\mathbf{w}^\star$ | the minimiser of a loss; $\mathbf{e} = \mathbf{w} - \mathbf{w}^\star$ is the error from it (Chapters 8, 9) |
| $\kappa$ | the condition number of a bowl: largest curvature divided by the smallest (Chapter 8) |
| $\kappa_f(x)$ | the relative condition number of a function at $x$, $\lvert x f'(x)/f(x)\rvert$ (Chapter 15) |
| $S_n^{\text{left}}$, $T_n$ | the left Riemann sum and the trapezoid rule with $n$ panels (Chapter 9) |
| $x(t)$, $\mathbf{x}(t)$ | the state of a differential equation at time $t$ (Chapter 9; $t$ is time here) |
| $O(\cdot)$, $\Theta(\cdot)$, $\Omega(\cdot)$ | growth rates of a cost (Chapter 16) |
| $\oplus$, $\gg$ | bitwise exclusive-or, and shift right (the pseudocode's `xor`, `>>`) |
| $\gamma$ | a decay factor, e.g. in a learning-rate schedule (Chapter 2); in Volume II, Chapter 2, the factor $\gamma = \max_i \lvert 1 - \eta\lambda_i \rvert$ by which gradient descent on a quadratic shrinks the error per step |

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
| $\mathrm{PPL}$ | perplexity, $\exp(H(p, q))$ with the entropy in nats (Chapter 13) |
| $\hat{p}$, $\hat\theta$ | an estimate computed from data (the hat marks an estimate) |
| $\alpha$ | significance level of a test, or confidence level $1 - \alpha$ |
| $\mathcal{N}(\mu, \sigma^2)$ | Gaussian with mean $\mu$, variance $\sigma^2$ |
| $\phi(z)$, $\Phi(z)$ | density and cumulative distribution function of $\mathcal{N}(0, 1)$ |
| $\chi^2$ | the chi-square statistic or distribution |
| $X^2$, $O_j$, $E$ | Pearson's chi-square statistic $X^2 = \sum_j (O_j - E)^2/E$ of the observed counts $O_j$ against the expected count $E$ (Chapter 17) |
| $H(p)$ | entropy |
| $H(p, q)$ | cross-entropy of $q$ relative to $p$ |
| $H(X \mid Y)$, $I(X; Y)$ | conditional entropy and mutual information |
| $H(X, Y)$ | joint entropy of two random variables $X$, $Y$ (the entropy of their joint distribution; two distributions $p, q$ as arguments, $H(p, q)$, mean cross-entropy) |
| $D_{\mathrm{KL}}(p \,\Vert\, q)$ | Kullback-Leibler divergence |
| $\mathrm{Lik}(\theta)$, $\mathrm{NLL}(\theta)$ | the likelihood of the parameters $\theta$ for the observed data, $p(\mathcal{D}\mid\theta)$, and its negative natural logarithm, the negative log-likelihood (Volume II, Chapter 4) |
| $\sigma_w$, $\beta_w$ | the standard deviation of a Gaussian prior on each weight, and the scale of a Laplace prior $\frac{1}{2\beta_w}e^{-\lvert w\rvert/\beta_w}$ (Volume II, Chapter 4; $\beta_w$ with its subscript is unrelated to the moment coefficients of the optimisers of Volume III) |

## Learning

| Symbol | Meaning |
|---|---|
| $\theta$ | all parameters of a model, as one long vector |
| $f_\theta$ | a model with parameters $\theta$ |
| $\mathcal{D} = \{(\mathbf{x}_n, y_n)\}_{n=1}^N$ | a dataset of $N$ examples |
| $\hat{y}$ | a prediction (the hat marks an estimate) |
| $L$, $\mathcal{L}$ | loss for one example, loss averaged over a batch or dataset |
| $\mathcal{H}$ | the hypothesis space: all functions $f_\theta$ reached by varying $\theta$ (Volume II, Chapter 1) |
| $E(\theta)$ | the average absolute error of $f_\theta$ on the known rows, with an argument (Volume II, Chapter 1; Chapter 2 of that volume replaces it by the loss $\mathcal{L}$) |
| $\ell(\hat y, y)$, $\mathrm{MSE}$ | the loss of one prediction against its label, so $L = \ell(f_\theta(\mathbf{x}), y)$, and the mean squared error $\frac{1}{N}\sum_n (\hat y_n - y_n)^2$ (Volume II, Chapter 2) |
| $\mathbf{X}$, $\mathbf{y}$, $\mathbf{w}$, $d$ | the feature matrix of shape $[N, d]$ (with a first column of ones when there is an intercept), the label vector of shape $[N]$, the weights of a model that is linear in its parameters ($\theta = \mathbf{w}$ there) and the number of features (Volume II) |
| $\tau$, $\tau_g$, $\tau_\ell$ | tolerance of a stopping rule: in general, on the gradient norm, on the change of the loss (Volume II, Chapter 2) |
| $\lambda$, $J(\mathbf{w})$, $S(a, \lambda)$ | in Volume II, Chapter 4: the strength of a penalty added to a loss, the penalised objective ($J_{\mathrm{L2}}$ with $\frac{\lambda}{2}\lVert\mathbf{w}\rVert^2$, $J_{\mathrm{L1}}$ with $\lambda\lVert\mathbf{w}\rVert_1$), and the soft-threshold function $S(a, \lambda) = \mathrm{sign}(a)\max(\lvert a\rvert - \lambda, 0)$ (the pseudocode writes it out, because $\mathrm{sign}(0) = +1$ and the product is still $0$ there) |
| $\eta$ | learning rate; $\eta_{\max}$, $\eta_{\min}$ are the largest and smallest rate of a schedule |
| $t$ | training step, or time in a sequence or a diffusion process |
| $\sigma(x)$ | the logistic sigmoid $1/(1+e^{-x})$ (always written with an argument) |
| $z$, $\mathrm{logit}(p)$ | in Volume II, Chapter 3, the logit of an example, $z = \mathbf{w}\cdot\mathbf{x}$ (with the intercept inside $\mathbf{w}$), and the function $\mathrm{logit}(p) = \log\frac{p}{1-p}$ that gives the logit of a probability; $\sigma$ is its inverse |
| $\mathrm{softplus}(z)$ | $\log(1 + e^z)$, a smooth version of $\max(z, 0)$ (Volume II, Chapter 3) |
| $\hat{p}$, $\mathbf{S}$ | the predicted probability $\hat{p} = \sigma(z)$ of the label $1$, and the diagonal matrix $\mathrm{diag}(\hat{p}_n(1 - \hat{p}_n))$ in the Hessian of the logistic loss (Volume II, Chapter 3) |
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
| hat | on a vector, $\hat{\mathbf{u}}$ is a unit vector; on a scalar or parameter ($\hat y$, $\hat p$) it is an estimate. The one vector exception is $\hat{\mathbf{y}} = \mathbf{X}\mathbf{w}$ (Volume II), the vector of predictions, an estimate of the labels $\mathbf{y}$ |
| $\varepsilon$ | machine epsilon (Chapters 1 and 7); in Chapters 11 and 12 a tolerance in the law of large numbers, and the standard Gaussian noise vector of a diffusion step. In Volume II, Chapter 2, the Gaussian noise added to a label; the tolerances of its stopping rules are $\tau$. Finite-difference and gradient-descent steps are written $h$ and $\eta$ |
| $\varphi$, $\phi$ | in Chapters 11 and 12 the standard Gaussian density (written $\varphi$ or $\phi$ alike); in Chapter 8 $\varphi(t)$ is the loss along a line |
| $\lambda$ | an eigenvalue (Chapter 6); in Chapters 8 and 9 the curvature of the loss along an eigenvector of the Hessian, which is the eigenvalue; in Volume II, Chapter 4, the strength of a regularization penalty (written $\lambda$ there and never an eigenvalue) |
| $u$ | the unit round-off $2^{-p}$ (Chapter 15); in Chapter 17 a uniform random number in $[0, 1)$, written $u$, $u_1$, $u_2$. The Chapter 17 constant of the SplitMix64 generator is written `GAMMA`, never $\gamma$ |

## Units

Sizes of memory are in bytes. Binary prefixes: 1 KiB = 1024 bytes,
1 MiB = 1024 KiB, 1 GiB = 1024 MiB. Decimal prefixes: 1 KB = $10^3$ bytes,
1 MB = $10^6$ bytes, 1 GB = $10^9$ bytes, so 1 GiB is about 7.4 percent
more than 1 GB. Counts of parameters use decimal
prefixes: a 7B model has about $7 \times 10^9$ parameters. Rates of
arithmetic are in floating-point operations per second (FLOP/s), counting
a multiply and an add as two operations.
