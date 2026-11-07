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
| $\tilde{\mathbf{x}}$, $\tilde{\mathbf{W}}$ | in Chapter 4, the input with a 1 appended, $(\mathbf{x}, 1)$, and the weight matrix with the bias as an extra last column, $[\mathbf{W}\,|\,\mathbf{b}]$; Volume III, Chapter 1 uses $\tilde{\mathbf{x}}$ again for the input with a 1 appended and $\tilde{\mathbf{w}} = (\mathbf{w}, b)$ for the weights with the bias appended |
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
| $\epsilon$ | in Chapters 6, 11, 13 and 15 (and, in Volume II, Chapter 7, the small floor added to an eigenvalue before whitening), a small amount or a relative perturbation (as in $x(1 + \epsilon)$ or a probability shifted by $\epsilon$); $\varepsilon$ is reserved for machine epsilon and the tolerances listed in "Symbols with more than one meaning" |
| $\beta$ | in Chapters 6 and 14 only, a rotation angle (as $\alpha$ is); in Volume II the scale $\beta_w$ of a prior |
| $\mathrm{atol}$, $\mathrm{rtol}$ | absolute and relative tolerance of `close` (the pseudocode page) |
| $\mathrm{ana}$, $\mathrm{num}$, $\mathrm{err}_{\mathrm{rel}}$, $\mathrm{rnd}_L$ | in Volume III, Chapter 6, the gradient entry (or directional derivative) from the backward pass and the one from a finite difference of the same forward pass, their relative error $\lvert\mathrm{ana} - \mathrm{num}\rvert / \max(\lvert\mathrm{ana}\rvert, \lvert\mathrm{num}\rvert)$, and the absolute error of one computed loss from rounding; the gate $\lvert\mathrm{ana} - \mathrm{num}\rvert \le \mathrm{atol} + \mathrm{rtol}\max(\lvert\mathrm{ana}\rvert, \lvert\mathrm{num}\rvert)$ is symmetric, unlike `close`, which scales by its second argument only |
| $\varepsilon$, $\varepsilon_{\mathrm{mach}}$ | machine epsilon: the gap between 1 and the next larger floating-point number ($2^{1-p}$, which is $2^{-52}$ in 64 bits); Chapter 15 writes $\varepsilon_{\mathrm{mach}}$ |
| $t$ | time, a training step or the time argument of a function (Chapters 7, 9; Volume II, Chapter 2); in Volume II, Chapters 3 and 6 the probability threshold of a classifier, with $\mathrm{logit}(t)$ the matching threshold on the logit |
| $u$ | the unit round-off $2^{-p} = \varepsilon_{\mathrm{mach}}/2$: the largest relative error of rounding one number (Chapter 15) |
| $\gamma$ | a decay factor (Chapter 2) or, in Volume II, the contraction factor of gradient descent on a quadratic (Chapter 2) and the width of the Gaussian kernel (Chapter 7); in Volume III, Chapter 1, the margin of a separator. The text says which |
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
| $D_i$, $\bar{D}$, $s_D$ | the difference between two systems' scores on example $i$ (second minus first), its sample mean (the gain) and its sample standard deviation, divisor $n - 1$ (Volume I, chapter 12; Volume II, chapter 8, where $\bar{D}/s_D$ is the standardised effect) |
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
| $\mathcal{D}_{\mathrm{train}}$, $\mathcal{D}_{\mathrm{val}}$, $\mathcal{D}_{\mathrm{test}}$ | the training, validation and test sets: disjoint parts of the available examples, used to choose parameters, to choose between models, and to report one final score (Volume II, Chapter 5) |
| $\hat f_{\mathcal{D}}$, $\bar f(\mathbf{x})$ | the model chosen by fitting to the dataset $\mathcal{D}$, and the average of its prediction at $\mathbf{x}$ over many datasets drawn from the same source, $\bar f(\mathbf{x}) = \mathbb{E}_{\mathcal{D}}[\hat f_{\mathcal{D}}(\mathbf{x})]$ (Volume II, Chapter 5) |
| $\mathcal{R}(f)$ | the expected loss of a model $f$ on a new example drawn from the source, $\mathbb{E}_{(\mathbf{x}, y) \sim p}[\ell(f(\mathbf{x}), y)]$ (Volume II, Chapter 5) |
| $k$, $\mathrm{CV}_k$ | in Volume II, Chapter 5, the number of folds of a cross-validation and the cross-validation estimate of the expected loss from $k$ folds |
| $\eta$ | learning rate; $\eta_{\max}$, $\eta_{\min}$ are the largest and smallest rate of a schedule |
| $t$ | training step, or time in a sequence or a diffusion process |
| $\sigma(x)$ | the logistic sigmoid $1/(1+e^{-x})$ (always written with an argument) |
| $z$, $\mathrm{logit}(p)$ | in Volume II, Chapter 3, the logit of an example, $z = \mathbf{w}\cdot\mathbf{x}$ (with the intercept inside $\mathbf{w}$), and the function $\mathrm{logit}(p) = \log\frac{p}{1-p}$ that gives the logit of a probability; $\sigma$ is its inverse |
| $\mathrm{softplus}(z)$ | $\log(1 + e^z)$, a smooth version of $\max(z, 0)$ (Volume II, Chapter 3) |
| $\hat{p}$, $\mathbf{S}$ | the predicted probability $\hat{p} = \sigma(z)$ of the label $1$, and the diagonal matrix $\mathrm{diag}(\hat{p}_n(1 - \hat{p}_n))$ in the Hessian of the logistic loss (Volume II, Chapter 3) |
| $\mathrm{softmax}(\mathbf{z})_i$ | $e^{z_i} / \sum_j e^{z_j}$ |
| $\mathrm{TP}$, $\mathrm{FP}$, $\mathrm{FN}$, $\mathrm{TN}$ | in Volume II, Chapter 6, the numbers of true positives, false positives, false negatives and true negatives among the decisions of a classifier at one threshold $t$ (a positive is a decision for the label $1$) |
| $n_+$, $n_-$, $z_i^{+}$, $z_j^{-}$ | the numbers of examples with the label $1$ and with the label $0$, and the scores (usually logits) of the $i$-th of the first and the $j$-th of the second (Volume II, Chapter 6) |
| $\mathrm{TPR}$, $\mathrm{FPR}$, $F_1$, $\mathrm{AUC}$, $\mathrm{AP}$ | true positive rate $\mathrm{TP}/n_+$ (the recall), false positive rate $\mathrm{FP}/n_-$, the F1 score $2\mathrm{TP}/(2\mathrm{TP} + \mathrm{FP} + \mathrm{FN})$, the area under the ROC curve, and the average precision, the step-wise area under the precision-recall curve (Volume II, Chapter 6) |
| $\mathrm{ECE}$ | the expected calibration error, $\sum_b \frac{n_b}{N}\lvert\mathrm{freq}_b - \mathrm{conf}_b\rvert$ over bins $b$ of the predicted probability (Volume II, Chapter 6) |
| $a$, $b$, $c_{\mathrm{FP}}$, $c_{\mathrm{FN}}$ | in Volume II, Chapter 6, the scale and the shift of a recalibrated probability $\sigma(a z + b)$ (the temperature is $1/a$), and the costs of one false positive and of one false negative |
| $k$, $\mathcal{N}_k(\mathbf{x})$ | in Volume II, Chapter 7, the number of neighbours consulted by nearest-neighbour classification or the number of groups of k-means (not the number of folds $k$ of Chapter 5), and the set of the $k$ training rows nearest to $\mathbf{x}$ |
| $\boldsymbol{\mu}_j$, $g_n$, $\mathcal{J}$ | in Volume II, Chapter 7, the centre of group $j$, the group (0-indexed in pseudocode) that row $n$ is assigned to, and the k-means objective, the sum over rows of the squared distance to the assigned centre ($\mathcal{J}$ has no weights as an argument, unlike $J(\mathbf{w})$ of Chapter 4) |
| $\mathbf{C}$, $\mathbf{u}_j$, $\mathbf{X}_c$ | in Volume II, Chapter 7, the covariance matrix $\frac{1}{N}\mathbf{X}_c^\top\mathbf{X}_c$ of the examples (divisor $N$, written $\mathbf{C}$ because $\boldsymbol{\Sigma}$ is the matrix of singular values in the same chapter), the $j$-th principal direction (a unit eigenvector of $\mathbf{C}$ with eigenvalue $\lambda_j$, largest first) and the feature matrix with the mean of each column subtracted |
| $G(\cdot)$, $\mathrm{IG}$ | in Volume II, Chapter 7, the Gini impurity $1 - \sum_c p_c^2$ of the class shares $p_c$ of a node, and the information gain of a split: the entropy of the node minus the size-weighted entropy of its two children |
| $M$, $q_n$, $v_m$, $h_m$ | in Volume II, Chapter 7, the number of trees or boosting rounds, the weight of example $n$ (the weights sum to one), the vote weight of round $m$ and the weak classifier of round $m$ |
| $\boldsymbol{\psi}(\mathbf{x})$, $\mathcal{K}(\mathbf{x}, \mathbf{x}')$, $\mathbf{K}$, $\gamma$, $\mathbf{c}$ | in Volume II, Chapter 7, a feature map, the kernel $\boldsymbol{\psi}(\mathbf{x})\cdot\boldsymbol{\psi}(\mathbf{x}')$ it defines, the Gram matrix $K_{nm} = \mathcal{K}(\mathbf{x}_n, \mathbf{x}_m)$ of the training rows, the width parameter of the Gaussian kernel $\exp(-\gamma\lVert\mathbf{x} - \mathbf{x}'\rVert^2)$ (not the decay factor of Chapter 2), and the dual coefficients of kernel ridge regression |
| $\nu_n$, $\alpha$ | in Volume II, Chapter 7, the multiplier of training row $n$ in the support vector machine (the literature writes $\alpha_n$), and the pseudo-count that Laplace smoothing adds to every count (not a significance level) |
| $g$, $\mathrm{step}(z)$, $b$ | in Volume III, Chapter 1, the activation function of a neuron, $y = g(\mathbf{w}\cdot\mathbf{x} + b)$; the step function, $\mathrm{step}(z) = 1$ for $z \ge 0$ and $0$ otherwise (so $\mathrm{step}(0) = 1$, the convention of the threshold rule of Volume II, Chapter 3); and the bias $b$ of one neuron (not the shift $b$ of a recalibrated probability, Volume II, Chapter 6) |
| $y \in \{-1, +1\}$, $t \in \{0, 1\}$ | in Volume III, Chapter 1, the label of an example in the perceptron's mistake rule and its proof, and the same label written as a target $0$ or $1$ when the rule is compared with logistic regression, $t = (y + 1)/2$ |
| $R$, $\gamma$, $\mathbf{s}$, $k$ | in Volume III, Chapter 1, the largest length $\lVert\tilde{\mathbf{x}}_n\rVert$ of an input with a 1 appended, the margin of a separator ($\gamma = \min_n y_n\,\mathbf{s}\cdot\tilde{\mathbf{x}}_n$ for a unit vector $\mathbf{s}$; not the Gaussian-kernel width of Volume II, Chapter 7), a unit vector in the space of $\tilde{\mathbf{w}}$ that separates the data, and the number of updates made so far (not a number of folds or neighbours) |
| $B$, $D$, $H$, $O$, $\mathbf{W}_\ell$, $\mathbf{b}_\ell$, $\mathbf{Z}_\ell$, $\mathbf{A}_\ell$ | in Volume III, Chapters 2 and 4, the batch size, input width, hidden width and output width; for layer $\ell = 1, 2, \dots$ of a network, its weight matrix ($[\text{outputs}, \text{inputs}]$), bias vector, pre-activation ($\mathbf{Z}_\ell = \mathbf{A}_{\ell-1}\mathbf{W}_\ell^\top + \mathbf{1}\mathbf{b}_\ell^\top$) and activation ($\mathbf{A}_\ell = g(\mathbf{Z}_\ell)$, with $\mathbf{A}_0 = \mathbf{X}$). In Chapter 2 the Volume II matrices $\mathbf{X}$ keep their row-per-example layout |
| $k$, $x_j$, $s_j$, $c_j$, $u$ | in Volume III, Chapter 2, the number of ReLU units (equally, of intervals) of the interpolating network, the knots $x_j = j/k$, the slope $s_j$ of the curve between $x_j$ and $x_{j+1}$, the output weight $c_j$ of unit $j$, and the number of units of a one-hidden-layer network in the piece count (here $k$ is not the update count of Chapter 1, and $D$, the input width, is the only dimension) |
| $\mathrm{gelu}$, $\mathrm{silu}$, $\mathrm{tanh}$, $\mathrm{tent}$, $\mathrm{tent}^{\circ d}$ | in Volume III, Chapter 2, the Gaussian-error linear unit $x\,\Phi(x)$, the sigmoid linear unit $x\,\sigma(x)$, the hyperbolic tangent, the tent map $\mathrm{tent}(x) = 2\min(x, 1 - x)$ on $[0, 1]$, and its $d$-fold composition $\mathrm{tent}^{\circ d} = \mathrm{tent}\circ\cdots\circ\mathrm{tent}$ ($d$ applications, the sawtooth of the depth example; the capital $T$ of Chapter 4 is a target matrix and is not used here) |
| $C$, $\mathbf{z}$, $\mathbf{p}$, $c$, $\mathbf{t}$, $\mathbf{t}'$ | in Volume III, Chapter 3, the number of classes, the vector of $C$ logits of one example (the italic $z$ of Volume II is one logit), the probability vector $\mathbf{p} = \mathrm{softmax}(\mathbf{z})$, the index of the true class (not the output weight $c_j$ of Chapter 2), the one-hot target $\mathbf{t} = \mathbf{e}_c$ and the smoothed target $\mathbf{t}' = (1-\epsilon_{\mathrm{ls}})\mathbf{t} + (\epsilon_{\mathrm{ls}}/C)\mathbf{1}$ |
| $d\mathbf{Y}$, $d\mathbf{W}$, $d\mathbf{b}$, $dy_i$, $K$, $B_j$ | in Volume III, Chapters 4 and 5, $d\mathbf{Y}$ is the matrix of all $\partial L/\partial Y_{bo}$ (the bar $\bar{Y}$ of Chapter 4 written with the shape of $\mathbf{Y}$; never a differential), and likewise $d\mathbf{W}$, $d\mathbf{b}$ and, for an entry, $dy_i$; in Chapter 5 only, $K$ is the number of micro-batches of a split batch and $B_j$ the number of examples of micro-batch $j$ (the $k$ of a matrix product's inner dimension and the $K$ of Volume II, Chapter 1 are elsewhere) |
| $\epsilon_{\mathrm{ls}}$, $a$, $\mathbf{J}$, $L$ | in Volume III, Chapter 3, the weight of the uniform part of a smoothed target (a small amount, so $\epsilon$ with a subscript, not the machine epsilon $\varepsilon$), the scale of the logits in $\mathrm{softmax}(a\mathbf{z})$ (the $a$ of Volume II, Chapter 6), the $C \times C$ Jacobian $\mathbf{J} = \mathrm{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top$ of the softmax, and the loss $L = \mathrm{lse}(\mathbf{z}) - z_c$ of one example |
| $\mathrm{odds}(p)$, $\lambda_{\mathrm{cap}}$, $\mathbf{v}$ | in Volume II, Chapter 3, the odds $p/(1-p)$ of a probability $p$; the curvature cap $\frac{1}{4}\lambda_{\max}(\frac{1}{N}\mathbf{X}^\top\mathbf{X})$, an upper bound on the largest eigenvalue of the Hessian of the logistic loss (written with a subscript label because $L$ is a loss); and a direction, any vector with the shape of $\mathbf{w}$, in $\mathbf{v}^\top\mathbf{H}\mathbf{v}$ |
| $\mathrm{BCE}$, $\mathrm{SE}$, $m$, $\beta_{\mathrm{pref}}$ | in Volume II, Chapter 3, table abbreviations for the binary cross-entropy and the squared error of one example; and, in the preference-loss example, the margin $m$ (so that the loss is $\mathrm{softplus}(-m)$) and the scale $\beta_{\mathrm{pref}}$ of the log-probability differences (the word `count` and the names `lp_chosen`, `lp_rejected` and their `_ref` versions are labels, as above) |
| $v$, $x$ | in Volume II, Chapter 5, the voltage of the sensor, $v \in [0.4, 2.8]$ V, and the feature $x = (v - 1.6)/1.2$ made from it. (A variance is written $\mathrm{Var}$ there, never $v$; the vote weight $v_m$ of Chapter 7 carries an index) |
| $C$, $G$, $p_1$ | in Volume II, Chapter 5, the number of candidates (models, penalties or coin-flippers) among which one is chosen, the number of groups (subjects) in a group split, and the share of class 1 in the data in use (Chapter 5, label shift). The italic $C$ is a count; bold $\mathbf{C}$ is the covariance matrix of Chapter 7, and $G(\cdot)$ with an argument is the Gini impurity of Chapter 7 |
| $F(m)$, $\Delta_{\mathrm{emb}}$, $x_0$ | in Volume II, Chapter 5, the cumulative distribution function of the number of correct answers of one coin-flipping candidate, the number of steps of an embargo between a training period and the period scored (not the step size $h$), and the threshold on the feature $x$ of a classifier that answers class 1 when $x > x_0$ (the threshold $t$ of Chapter 6 is on a probability or a score) |
| $\mathrm{score}_j$ | in Volume II, Chapter 5, the loss of the model fitted without fold $j$, measured on fold $j$; $\mathrm{CV}_k$ is their mean |
| $w_0$, $w_1$ | in Volume II, Chapter 2, the intercept and the slope of a line as the two entries of $\mathbf{w}$, intercept first because the first column of the design matrix is ones (the same line is $b$ and $w$ in Chapter 1 and in the tables, which print the pair as (slope, intercept)) |
| $r_n$, $\mathbf{r}$ | in Volume II, Chapter 2 and after, the residual of row $n$, $r_n = \hat y_n - y_n$ (prediction minus label), and the residual vector $\mathbf{r} = \mathbf{X}\mathbf{w} - \mathbf{y}$; the residual of a projection in Volume I, Chapter 3 has the opposite sign, which changes no length and no right angle |
| $\bar v$, $\bar y$, $s$, $S_{vv}$, $S_{vy}$ | in Volume II, Chapter 2, the sample means of the voltages and of the labels, the sample standard deviation of the voltages (divisor $N - 1$), and the unnormalised sums $S_{vv} = \sum_n (v_n - \bar v)^2$ and $S_{vy} = \sum_n (v_n - \bar v)(y_n - \bar y)$ whose ratio $S_{vy}/S_{vv}$ is the least-squares slope |
| $\mathbf{A}$, $\mathbf{c}$ | in the derivation of the normal equations (Volume II, Chapter 2) only, $\mathbf{A} = \mathbf{X}^\top\mathbf{X}$ and $\mathbf{c} = \mathbf{X}^\top\mathbf{y}$, the matrix and the right-hand side passed to the solver (not the dual coefficients $\mathbf{c}$ of Chapter 7) |
| $\mathcal{B}_t$, $\mathbf{g}_t$, $\nabla\ell_n$, $\sigma_g^2$ | in Volume III, Chapter 7, the set of $B$ example indices of the minibatch of step $t$; the minibatch gradient $\mathbf{g}_t = \frac{1}{B}\sum_{n\in\mathcal{B}_t}\nabla\ell_n(\theta_{t-1})$ (bold with a step subscript, never the activation function $g$ of Chapter 1); the gradient of the loss of example $n$; and the total variance $\sigma_g^2 = \mathbb{E}\lVert\nabla\ell_n - \nabla\mathcal{L}\rVert^2$ of one example's gradient (the sum of its $P$ component variances) |
| $\mathbf{m}_t$, $\mathbf{v}_t$, $\hat{\mathbf{m}}_t$, $\hat{\mathbf{v}}_t$, $\beta$, $\beta_1$, $\beta_2$, $\epsilon_{\mathrm{den}}$ | in Volume III, Chapter 7, the momentum buffer (sum form, $\mathbf{m}_t = \beta\mathbf{m}_{t-1} + \mathbf{g}_t$) or the exponential average of the gradient (average form, with $1 - \beta_1$ on the new gradient); the exponential average of the squared gradient, elementwise; their bias-corrected versions ($\hat{\mathbf{m}}_t = \mathbf{m}_t/(1-\beta_1^t)$); the decay coefficients of those averages in $[0, 1)$; and the small floor $\epsilon_{\mathrm{den}}$ (a subscripted $\epsilon$, like $\epsilon_{\mathrm{ls}}$, not the machine epsilon $\varepsilon$) added to $\sqrt{\hat{\mathbf{v}}_t}$ in a denominator. The $\beta_w$ of Volume II, Chapter 4 is unrelated |
| $a$, $V$, $\lambda$ (Chapter 7) | in Volume III, Chapter 7 only, $a > 0$ is a constant that multiplies every gradient (the scale of the gradient, in the section on scale invariance; the $a$ of Chapter 3 is a logit scale), $V = \mathbb{E}[e^2]$ is the stationary mean square error of SGD on one weight, and $\lambda$ without a subscript is the strength of the L2 penalty or of the decoupled weight decay, while the curvatures of the section on momentum keep their subscripts, $\lambda_{\max}$, $\lambda_{\min}$ |
| $\eta_t$, $W$, $T$, $T_{\mathrm{hold}}$, $T_{\mathrm{cool}}$, $u$, $c(u)$ | in Volume III, Chapter 8, the learning rate used at step $t$ (0-indexed; Volume I, Chapter 2 writes the same function); the number of warmup steps; the total number of steps of the run (not a sequence length); the steps held at the peak and the steps of the final decay (so that $T = W + T_{\mathrm{hold}} + T_{\mathrm{cool}}$); the fraction $u = (t - W - T_{\mathrm{hold}})/T_{\mathrm{cool}}$ of the decay that has been done, and the cosine decay factor $c(u) = \tfrac12(1 + \cos\pi u)$ |
| $\tau_{\mathrm{clip}}$, $c_t$, $\epsilon_{\mathrm{clip}}$, $\lVert\mathbf{g}_t\rVert$ | in Volume III, Chapter 8, the clip threshold on the global gradient norm; the clip coefficient $c_t = \min(1, \tau_{\mathrm{clip}}/(\lVert\mathbf{g}_t\rVert + \epsilon_{\mathrm{clip}}))$ by which the whole gradient is multiplied (a scalar, unrelated to the coefficients $c_j$ of Volume II, Chapter 4); the small constant $\epsilon_{\mathrm{clip}} = 10^{-6}$ that keeps the quotient finite; and the global gradient norm, the Euclidean norm of the gradient of every parameter of the model taken as one vector |
| $\lambda_{\max}$, $\lambda_{\min}$, $\eta^\star$, $\mathbf{u}_i$, $\xi_i$ | in Volume II, Chapter 2, the largest and smallest eigenvalue of the Hessian $\mathbf{H} = \frac{2}{N}\mathbf{X}^\top\mathbf{X}$ of the mean squared error, the best learning rate $2/(\lambda_{\max} + \lambda_{\min})$, the unit eigenvectors of $\mathbf{H}$ and the coefficients of the error $\mathbf{e}$ along them, $\mathbf{e} = \sum_i \xi_i\mathbf{u}_i$ |
| $R^2$, $\mathrm{SS}_{\mathrm{res}}$, $\mathrm{SS}_{\mathrm{tot}}$ | in Volume II, Chapter 2, the sum of the squared residuals of a fitted line, the sum of the squared deviations of the labels from their mean, and the quality number $R^2 = 1 - \mathrm{SS}_{\mathrm{res}}/\mathrm{SS}_{\mathrm{tot}}$ (1 for a perfect line, 0 for a line no better than the mean) |
| $\mathcal{L}(n_{\mathrm{p}})$, $\mathcal{L}_\infty$, $\zeta$, $\alpha$ | in Volume II, Chapter 2, the power law of loss against model size $n_{\mathrm{p}}$ (a number of parameters): $\mathcal{L}(n_{\mathrm{p}}) = \mathcal{L}_\infty + \zeta\, n_{\mathrm{p}}^{-\alpha}$ with floor $\mathcal{L}_\infty$, scale $\zeta$ and exponent $\alpha$ |
| $r$, $\mathrm{score}_j$, $\mathbf{x}_{\mathrm{rec}}$, $\mathbf{Q}$ | in Volume II, Chapter 7, the number of principal directions kept, the score $\mathrm{score}_j = \mathbf{u}_j\cdot(\mathbf{x} - \bar{\mathbf{x}})$ of a row on the $j$-th direction (a scalar, whereas $\mathbf{u}_j$ in bold is the direction), the rank-$r$ reconstruction $\mathbf{x}_{\mathrm{rec}} = \bar{\mathbf{x}} + \sum_{j \le r}\mathrm{score}_j\,\mathbf{u}_j$ (written without a hat, because a hat on a vector marks a unit vector), and the orthogonal matrix whose columns are the principal directions in the whitening paragraph (the $\mathbf{V}$ of the singular value decomposition of the same chapter) |
| $D(\mathbf{x})$, $n_L$, $n_R$, $n_c$, $p_c$, $W_c$, $n_{wc}$ | in Volume II, Chapter 7, the distance from a row to the nearest centre chosen so far in k-means++ (its square is `D2` in the pseudocode); the numbers of rows sent to the left and to the right by a split of a tree; the number of rows of class $c$ in a node and their share $p_c = n_c/n$; and, in naive Bayes, the total number $W_c$ of word occurrences in class $c$ and the count $n_{wc}$ of word $w$ in it |
| $\mathrm{err}_m$, $Z_m$, $F$, $\eta$ | in Volume II, Chapter 7, the weighted error of the stump of boosting round $m$ (written with the Iverson bracket $\mathbf{1}[\cdot]$ of Chapter 6 of this volume), the sum of the unnormalised new weights of that round, the running weighted sum of the models chosen so far ($F(\mathbf{x}) = \sum_m v_m h_m(\mathbf{x})$ in AdaBoost, whose class is $\mathrm{sign}(F(\mathbf{x}))$; the current prediction in gradient boosting), and the shrinkage, the learning rate of gradient boosting |
| $\boldsymbol{\Psi}$, $P$, $p$ | in Volume II, Chapter 7, the matrix of shape $[N, P]$ whose rows are the feature vectors $\boldsymbol{\psi}(\mathbf{x}_n)$, the number $P$ of features of a feature map, and (lower case) the degree of a polynomial feature map, which is the only place the chapter uses a bare $p$ |
| $\xi$, $P$, $R$, $W_+$ | in Volume II, Chapter 6, the base rate $\xi = P(y = 1)$ (the share of positive examples; not $\pi$, which stays the constant 3.14159..., and not the correlation $\rho$), the precision and the recall of $F_1 = 2PR/(P + R)$ (and $P_k$, $R_k$, their values after the $k$-th distinct score in the average precision), and the sum of the ranks of the positive examples in the rank form of the AUC (the Wilcoxon rank sum) |
| $B$, $n_b$, $\mathrm{freq}_b$, $\mathrm{conf}_b$ | in Volume II, Chapter 6, the number of bins of a reliability table, the number of examples in bin $b$, the share of them that are positive and their mean predicted probability |
| $\mathbf{1}[\cdot]$, $t^\star$ | in Volume II, Chapter 6, the Iverson bracket, 1 when the condition inside is true and 0 otherwise, and the cost-optimal threshold $c_{\mathrm{FP}}/(c_{\mathrm{FP}} + c_{\mathrm{FN}})$ on a calibrated probability |
| $\mathrm{MAE}$, $\bar{y}$, $\mathrm{SSE}$, $\mathrm{SST}$, $R^2$ | in Volume II, Chapter 6, the mean absolute error $\frac{1}{N}\sum_n \lvert\hat y_n - y_n\rvert$, the mean of the labels, the sum of squared errors $\sum_n (\hat y_n - y_n)^2$ (so $\mathrm{MSE} = \mathrm{SSE}/N$), the total sum of squares $\sum_n (y_n - \bar y)^2$ about the mean, and the coefficient of determination $R^2 = 1 - \mathrm{SSE}/\mathrm{SST}$ (the exponent 2 is part of the name; it is not $R$ squared, and $R$ here is unrelated to the recall $R$) |
| $T$ | in Volume II, Chapter 6, the temperature $T = 1/a$ of a recalibrated probability $\sigma(a z)$ (see "Symbols with more than one meaning") |
| $v$, $w$, $b$, $a$ | in Volume II, Chapter 1, the voltage of the sensor example (the one feature, so $x_n = v_n$), the slope and intercept of the line $wv + b$, and the coefficient of $v^2$ in the parabola $av^2 + wv + b$; from Chapter 2 on $w$ and $b$ are the weights and the intercept of a linear model |
| $m$, $k$, $K$, $\mathbf{x}_n$ | in Volume II, Chapter 1, the number of values per parameter of a search grid, the number of parameters (so a grid has $m^k$ settings), the number of centres in the grouping pseudocode, and the feature vector of example $n$ (a one-entry vector for the sensor) |
| $n$, $p_{11}$, $p_{10}$, $p_{01}$, $p_{00}$, $\mathrm{sd}[\cdot]$ | in Volume II, Chapter 8, $n$ is the number of questions (test examples, or experiments' draws) in one sample, while $N$ stays the number of examples of a dataset; $p_{11}$, $p_{10}$, $p_{01}$, $p_{00}$ are the probabilities that a question is answered right by both models, only by A, only by B, by neither (they sum to one); $\mathrm{sd}[X]$ is the standard deviation of a random quantity $X$ over repetitions (the square root of $\mathrm{Var}[X]$), written with the sample divisor $n - 1$ when computed from data |
| $k$, $m$, $r$ | in Volume II, Chapter 8, the number of tries of arm B, the number of tries of arm A (Exercise 3), and the number $r$ of components of an ablation, which gives $2^r$ cells (not the folds, neighbours or updates of the other chapters) |
| $h$, $t$, $\hat\theta_{\mathrm{ML}}$, $\hat\theta_{\mathrm{MAP}}$ | in Volume II, Chapter 4, the numbers of heads and tails among the flips of a coin (not the step $h$ of a finite difference or the training step $t$), and the maximum likelihood and maximum a posteriori estimates of a parameter ($\hat\mu$ and $\hat\sigma^2_{\mathrm{ML}}$ are the maximum likelihood mean and variance of a Gaussian) |
| $\mathrm{Beta}(a, b)$, $B(a, b)$ | in Volume II, Chapter 4, the Beta distribution on $(0, 1)$ with density $\theta^{a-1}(1-\theta)^{b-1}/B(a, b)$, and its normalising constant $B(a, b)$, which makes the area one; a prior $\mathrm{Beta}(a, b)$ acts like $a - 1$ extra heads and $b - 1$ extra tails |
| $Q$, $\mathbf{m}$ | in Volume II, Chapter 4, the sum of squared deviations $\sum_i (x_i - \hat\mu)^2$ from the sample mean, and the minimiser of the two-weight loss $\frac12(\mathbf{w}-\mathbf{m})^\top\mathbf{A}(\mathbf{w}-\mathbf{m})$ of the figure on penalties |
| $\rho_j$, $c_j$ | in Volume II, Chapter 4, in coordinate descent for the lasso, $\rho_j = \sum_n x_{nj}(y_n - \sum_{k\ne j}x_{nk}w_k)$ (not a correlation coefficient) and $c_j = \sum_n x_{nj}^2$ |
| $G$, $g$, $\mathrm{prof}_{gj}$, $\mathrm{eff}^{\mathrm{fail}}_g$, $\mathrm{eff}^{\mathrm{wear}}_g$ | in Volume II, Chapter 9, the number of pumps (groups of rows that share their source; $G(\cdot)$ with an argument is the Gini impurity of Chapter 7), the number of one pump, the profile of pump $g$ (the centre of its measurements, one number per feature $j$) and the two pump effects that its four rows share, on the failure logit and on the wear |
| $e_n$, $e^\star$, $z_{nj}$ | in Volume II, Chapter 9, the logit of row $n$ of a fitted model (written $z$ in Chapters 3 and 6), the logit of the law that generated the data, and the standardised value of feature $j$ in row $n$ (the z-score of Chapter 2) |
| $\bar y$, $t^\star$, $\mathrm{RMSE}$ | in Volume II, Chapter 9, the training share of rows with the label $1$ (the mean of the $0/1$ labels, as in Chapter 6), the cost-optimal probability threshold $c_{\mathrm{FP}}/(c_{\mathrm{FP}} + c_{\mathrm{FN}})$ of Chapter 6, and the root mean squared error $\sqrt{\mathrm{MSE}}$, in the units of the label |
| $n_{\mathrm{in}}$, $n_{\mathrm{out}}$, $\sigma_w^2$, $q_\ell$, $q^{\mathrm{b}}_\ell$, $\kappa_g$, $\Gamma$, $\omega$ | in Volume III, Chapter 9, the fan-in and fan-out of a layer (the number of inputs it sums over and its number of units), the variance of an initial weight (the $\sigma_w$ of Volume II, Chapter 4, there the width of a prior), the second moment $\mathbb{E}[z_\ell^2]$ of a pre-activation of layer $\ell$ and the second moment $\mathbb{E}[dz_\ell^2]$ of its gradient (the superscript b is for backward, never an exponent), the ratio $\kappa_g = \mathbb{E}[g(z)^2]/\mathbb{E}[z^2]$ of an activation (the subscript $g$ separates it from the condition number $\kappa$ of Volume I, Chapter 8), the factor $\Gamma = n_{\mathrm{in}}\sigma_w^2\kappa_g$ by which one layer multiplies a second moment, and the second-moment gain $\omega$ of a residual branch (not the $\gamma$ of the schedules) |
| $\mu$, $\sigma^2$, $\hat{x}_i$, $\sigma_\epsilon$, $\epsilon_{\mathrm{norm}}$, $\boldsymbol{\gamma}$, $\boldsymbol{\beta}$, $\beta_{\mathrm{run}}$, $G$ | in Volume III, Chapter 10, the mean and the variance (divisor $n$) of one group of $n$ values that a normalization layer treats together (not the parameters of a Gaussian); the normalized value $\hat x_i = (x_i - \mu)/\sigma_\epsilon$ (here the hat does not mark an estimate); $\sigma_\epsilon = \sqrt{\sigma^2 + \epsilon_{\mathrm{norm}}}$; the small floor $\epsilon_{\mathrm{norm}}$ (default $10^{-5}$) added under the square root (a subscripted $\epsilon$, not the machine epsilon $\varepsilon$); the learned gain and shift of the layer (bold vectors with one number per feature or channel; the scalar $\gamma$ and $\beta$ of schedules and optimizers are unrelated); the decay of the running averages of batch normalization (the weight of the old value, so $1 - \beta_{\mathrm{run}}$ goes on the new batch); and the number of channel groups of group normalization |
| $p_{\mathrm{drop}}$, $\mathbf{M}$, $s_{\mathrm{aug}}$, $E$, $P_{\mathrm{pat}}$, $t_{\mathrm{best}}$ | in Volume III, Chapter 11, the probability that dropout sets a unit to zero (the subscript keeps it apart from the bare $p$ of probabilities); the mask, an array with the shape of the activations whose entries are $1$ for a kept unit and $0$ for a dropped one, so that the layer sends $\mathbf{M}\odot\mathbf{A}/(1 - p_{\mathrm{drop}})$; the standard deviation of the noise that an augmentation adds to an input; the number of training steps between two evaluations on the validation set; the patience of early stopping (evaluations without improvement before stopping); and the step of the lowest validation loss. In the same chapter the ridge penalty keeps the bare $\lambda$ of Volume II, Chapter 4 and the curvature along an eigenvector of the Hessian is $\lambda_i$ with its index |
| $L_t$, $L^{\mathrm{sm}}_t$, $\beta_{\mathrm{sm}}$, $r_{\mathrm{upd}}$, $\mathbf{S}_\ell$, $\mathbf{J}_\ell$, $\alpha_{\mathrm{leak}}$, $\lVert\mathbf{A}\rVert_2$ | in Volume III, Chapter 12, the loss of the minibatch of step $t$ (not the per-example loss $\ell_n$ of Chapter 7), its bias-corrected exponential average and the decay of that average (the weight of the old value; not the $\beta$ of the optimizers, nor $\beta_{\mathrm{run}}$ of Chapter 10); the update ratio $r_{\mathrm{upd}} = \eta\lVert\mathbf{g}\rVert/\lVert\boldsymbol{\theta}\rVert$, the size of a step relative to the weights; the diagonal matrix $\mathbf{S}_\ell = \mathrm{diag}(g'(\mathbf{z}_\ell))$ of the slopes of layer $\ell$ and the Jacobian $\mathbf{J}_\ell = \mathbf{S}_\ell\mathbf{W}_\ell$ of its output with respect to its input (for a residual layer $\mathbf{J}_f$ is the Jacobian of the branch, and the layer's is $\mathbf{I} + \mathbf{J}_f$; the $\mathbf{J}$ of Chapter 3 is the softmax Jacobian); the slope $\alpha_{\mathrm{leak}} = 0.1$ of the leaky ReLU for negative inputs; and the matrix norm $\lVert\mathbf{A}\rVert_2$, the largest singular value of $\mathbf{A}$ (the longest stretch it gives a vector), which for a vector is the Euclidean norm |
| $\hat{\mathbf{x}}$, $\mathbf{z}$, $L_{\mathrm{rec}}$, $s_k$, $\mathrm{pat}_k$, $\mathrm{pos}_j$, $S$, $N_{\mathrm{ep}}$ | in Volume III, Chapter 13, an autoencoder's output (its reconstruction of the input $\mathbf{x}$; the hat marks the network's estimate), its bottleneck vector (the code), the reconstruction loss $\frac{1}{BD}\sum(\hat{x} - x)^2$; the strength $s_k$ of the $k$-th of the three causes of the sensor data, the pattern $\mathrm{pat}_k$ it multiplies and the position $\mathrm{pos}_j = j/15$ of sensor $j$; the number of steps in an epoch, $S = N/B$, and the number of epochs $N_{\mathrm{ep}}$ (the $E$ of Chapter 11 is a number of steps between evaluations) |
| $y_t$, $y^\ast_t$, $e_t$, $u_t$, $u^\ast_t$, $T_{\mathrm{p}}$, $G_{\mathrm{p}}$, $\Delta t$, $K_{\mathrm{p}}$, $K_{\mathrm{i}}$, $\zeta_{\mathrm{d}}$, $\omega_{\mathrm{n}}$, $\delta_u$ | in Volume III, Chapter 13, for the controlled plant: its measured level at step $t$, the setpoint, the error $e_t = y^\ast_t - y_t$, the command (clamped to $[-1, 1]$) and the expert's command; the plant's time constant and gain (the subscript keeps $T_{\mathrm{p}}$ apart from the length $T$ of a sequence) and the time step $\Delta t = 0.05$; the proportional and integral gains of the PI controller; the damping and natural frequency of its pole placement; and a command error of the learned controller |

## Computation

| Symbol | Meaning |
|---|---|
| $n_e$, $n_m$ | the number of exponent bits and of fraction bits of a floating-point format (Volume IV, chapter 1); the significand has $p = n_m + 1$ bits, so $\varepsilon = 2^{-n_m}$ and $u = 2^{-(n_m+1)}$ |
| $E$, $f$, $s$ | the stored exponent field (an unsigned integer), the stored fraction field and the sign bit of a floating-point code, with value $(-1)^s (1 + f/2^{n_m})\,2^{E - \mathrm{bias}}$ for a normal number, $\mathrm{bias} = 2^{n_e - 1} - 1$ (Volume IV, chapter 1) |
| $\Delta$, $q$, $q_0$ | the scale of an integer format, an integer code and the zero point, so that a stored code $q$ stands for $\Delta\,(q - q_0)$, with $q_0 = 0$ when there is no zero point (Volume IV, chapter 1) |
| $n_{\mathrm{lane}}$, $t_{\mathrm{lat}}$, $r_{\mathrm{iss}}$, $s_{\mathrm{line}}$, $n_{\mathrm{sets}}$, $n_{\mathrm{ways}}$ | in Volume IV, Chapter 2, the number of lanes of a vector register, the latency of an instruction in cycles, the number of instructions of a kind a core starts per cycle (the issue rate), the size of a cache line in bytes (64), and the number of sets and of ways per set of a cache, so that a cache of $C$ bytes has $C = n_{\mathrm{sets}}\,n_{\mathrm{ways}}\,s_{\mathrm{line}}$ |

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
| hat | on a vector, $\hat{\mathbf{u}}$ is a unit vector; on a scalar or parameter ($\hat y$, $\hat p$) it is an estimate.. The scalar $\hat x_i$ of Volume III, Chapter 10 is the normalized value of a normalization layer, not an estimate |
| $\varepsilon$ | machine epsilon (Chapters 1 and 7); in Chapters 11 and 12 a tolerance in the law of large numbers, and the standard Gaussian noise vector of a diffusion step. In Volume II, Chapter 2, the Gaussian noise added to a label; the tolerances of its stopping rules are $\tau$. Finite-difference and gradient-descent steps are written $h$ and $\eta$ |
| $\varphi$, $\phi$ | in Chapters 11 and 12 the standard Gaussian density (written $\varphi$ or $\phi$ alike); in Chapter 8 $\varphi(t)$ is the loss along a line |
| $\lambda$ | an eigenvalue (Chapter 6); in Chapters 8 and 9 the curvature of the loss along an eigenvector of the Hessian, which is the eigenvalue; in Volume II, Chapter 4, the strength of a regularization penalty (written $\lambda$ there and never an eigenvalue); in Volume III, Chapter 7 the same strength is the bare $\lambda$ of L2 penalties and decoupled weight decay, and the curvatures there always carry a subscript; in Chapter 7 of that volume both occur, the eigenvalue $\lambda_j$ of a covariance matrix with its index in the section on principal components, and the penalty $\lambda$ without an index in the sections on kernels |
| $a$, $b$ | in Volume II, Chapter 4, $a$ and $b$ are the two parameters of a Beta distribution in the section on the coin's prior, and $a$ is also the unpenalised value in the one-weight soft-threshold problem; the sections are apart and each says which is meant (Chapter 6 of that volume uses $a$, $b$ for a recalibration) |
| $u$ | the unit round-off $2^{-p}$ (Chapter 15); in Chapter 17 a uniform random number in $[0, 1)$, written $u$, $u_1$, $u_2$. The Chapter 17 constant of the SplitMix64 generator is written `GAMMA`, never $\gamma$. In Volume II, Chapter 7 the italic scalar $u$ does not occur: a direction is the bold $\mathbf{u}$ or $\mathbf{u}_j$, the scalar position of a row along it is $\mathrm{score}_j$, and the pseudocode of k-means++ names its random point `target` |
| $L$, $\ell$, $\lambda_{\mathrm{cap}}$ | $\ell(\hat y, y)$ is the loss of one example and $\mathcal{L}$ the mean loss over the data; italic $L$ in the Learning table is the loss of one example, and in Volume II, Chapter 3 it is not used for anything else: the curvature bound of the logistic loss is $\lambda_{\mathrm{cap}}$ |
| $S$ | bold $\mathbf{S}$ is a diagonal matrix (Volume II, Chapter 3); $S(a, \lambda)$ with arguments is the soft-threshold function; $S_{vv}$ and $S_{vy}$ with two subscripts are the sums of Volume II, Chapter 2; $S_n^{\text{left}}$ is a Riemann sum (Volume I, Chapter 9) |
| $\mathbf{e}$, $\mathbf{e}_j$ | bold $\mathbf{e}$ without an index is the error $\mathbf{w} - \mathbf{w}^\star$ of an iterate (and $\mathbf{e}_k$ with the step number $k$ as a subscript is that error after $k$ steps, Volume II, Chapter 2); $\mathbf{e}_j$ with an index $j$ of a coordinate is a unit vector |
| $\alpha$ | a rotation angle (Chapters 6 and 14); the exponent of a power law (Volume II, Chapter 2); the pseudo-count of Laplace smoothing (Volume II, Chapter 7); and the significance level of a test (Chapter 12), which never occurs in the same section as the others |
| $T$ | the number of positions of a sequence; in Volume II, Chapter 6 the temperature $1/a$ of a recalibrated probability $\sigma(a z)$, always introduced as "the temperature" |
| $R$ | the recall in $F_1 = 2PR/(P + R)$ and $R_k$ in the average precision (Volume II, Chapter 6); the largest input length of the perceptron (Volume III, Chapter 1). The two never occur in one argument |
| $\rho$, $\xi$ | $\rho$ is the correlation coefficient (and $\rho_j$ a local sum in Volume II, Chapter 4); the base rate $P(y = 1)$ of Volume II, Chapter 6 is $\xi$, so that it clashes neither with $\rho$ nor with the constant $\pi$ |
| $a$, $b$, $v$, $k$ | in Volume II, Chapter 1, $a$ and $b$ are parameters of the sensor lines and parabolas and $v$ is a voltage; $a$ and $b$ mean the recalibration scale and shift only in Chapter 6, $v_m$ with a subscript is a vote weight only in Chapter 7, and $k$ is a count of parameters in Chapter 1, of folds in Chapter 5 and of neighbours or groups in Chapter 7 |
| $C$, $G$, $f$ | the italic $C$ of Volume II, Chapter 5 is a number of candidates, the bold $\mathbf{C}$ of Chapter 7 a covariance matrix; $G$ without an argument is a number of groups (Chapter 5), $G(\cdot)$ with an argument the Gini impurity (Chapter 7); $f(\mathbf{x})$ without a hat or subscript is the true curve of the source in Chapter 5, $f_\theta$ a model, $\hat f_{\mathcal{D}}$ a fitted model |
| $\mathbf{z}$ in Volume III | bold $\mathbf{z}$ is the vector of logits in Chapter 3 and the code (the bottleneck vector of an autoencoder) in Chapter 13, written $\mathbf{z}$ there as in brain's source; each chapter says which, and they never meet |
| $z$ | the logit of an example (Volume II, Chapter 3, written $z$ or $z_i$ with the example); in Volume I, Chapter 12 and Volume II, Chapter 8 the test statistic $z = \bar{D}/(s_D/\sqrt{n})$ of a paired comparison, always written next to $\bar{D}$ or with the word statistic. The two never meet in one formula |
| $p$ | $p(x)$ with an argument is a probability or density; $p_{11}$ and the other $p$ with two digits are probabilities of the four outcomes of a question; bare $p$ in a table of tests (Volume I, Chapter 12; Volume II, Chapter 8) is the $p$-value. The number of features is $d$ in every pseudocode of Volume II, never $p$ |
| $M$ | italic $M$ is a number of trees or rounds (Volume II, Chapter 7) and, in Volume III, Chapter 10, the number of values $N\cdot H\cdot W$ of a normalization group; bold $\mathbf{M}$ in Volume III, Chapter 11 is the dropout mask. They never occur in one argument |
| $e$, $z$, $u$, $v$, $R$ in the project | in Volume II, Chapter 9, $z_{nj}$ is a standardised measurement, so the logit that Chapters 3 and 6 write $z$ is written $e_n$ there; the two pump effects are $\mathrm{eff}^{\mathrm{fail}}_g$ and $\mathrm{eff}^{\mathrm{wear}}_g$, not $u$ and $v$ (the unit round-off, the voltage); the number of bootstrap resamples is written out, not $R$ (the recall, the number of axes); $\lambda$ is the penalty and $\lambda_{\max}$, with its subscript, the largest eigenvalue of the design matrix, as in Chapter 7 |

## Units

Sizes of memory are in bytes. Binary prefixes: 1 KiB = 1024 bytes,
1 MiB = 1024 KiB, 1 GiB = 1024 MiB. Decimal prefixes: 1 KB = $10^3$ bytes,
1 MB = $10^6$ bytes, 1 GB = $10^9$ bytes, so 1 GiB is about 7.4 percent
more than 1 GB. Counts of parameters use decimal
prefixes: a 7B model has about $7 \times 10^9$ parameters. Rates of
arithmetic are in floating-point operations per second (FLOP/s), counting
a multiply and an add as two operations.
