<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Project: A Neural Network Library with Proven Gradients

*Can you build a small neural network library whose every gradient is checked, and train it to solve a problem?*

A neural network library can pass every test its author thought to write and still train on a wrong gradient for weeks, because the loss keeps falling and nothing looks broken. The only defence is to build the library so that its own checks would catch the fault. Thirteen chapters have supplied the parts: a layer, a loss, a backward pass written out for every operation, a way to prove the gradients right, five optimizers, a schedule and a clip, a starting point, normalization, regularization, a practice for reading a run that goes wrong, and two complete models. Whether they survive being put together is the question of this chapter. You will write, in a language of your choice, on top of the numerical library you built at the end of Volume I, a program with seven parts:

1. **Layers.** Objects with a forward pass that saves what the backward pass needs and a backward pass that accumulates gradients: linear, the activations tanh, ReLU, GELU and SiLU, softmax cross-entropy fused with its loss, the mean squared error, layer norm and RMS norm, dropout, a residual wrapper and a sequential container (chapters 2 to 5, 10, 11 and 12).
2. **A gradient checker.** Per-entry and directional, with a policy for the sizes of the test instance. Every layer passes it, and a layer with a planted fault does not (chapter 6).
3. **Optimizers.** SGD, momentum, Adam, AdamW and Lion with their state, the warmup-and-cosine schedule and global-norm clipping (chapters 7 and 8).
4. **Initialization.** He, Xavier and fan-in schemes in three shapes (chapter 9).
5. **A training loop.** Logging, early stopping and checkpointing, so that two runs with the same seeds agree exactly, and so does a run that is stopped, saved and resumed (chapters 11 to 13).
6. **Two models trained end to end.** A classifier with softmax cross-entropy on a three-class synthetic task, and the autoencoder of chapter 13.
7. **A debugging exercise.** A configuration that ships with three planted defects. You find each with the checklist of chapter 12 and fix it.

Five words recur, each defined by where it is used. A **fixture** is a small input fixed by a formula rather than by a generator, so that any language computes the identical numbers. A **probe** is a fixed array that stands in for the gradient arriving from above, so that a layer's output becomes a single number to differentiate. A **checkpoint** is the saved state of a training run, and the **patience** is the number of consecutive epochs without an improvement of the validation loss that the loop tolerates before it stops. A **directional** check compares the gradient with a finite difference along one direction instead of along each coordinate (chapter 6).

Nothing here is new. Every part was derived and worked by hand; the project asks whether the parts compose into a library you would trust, which is a different thing from following a derivation. The failures it is designed to expose are the ordinary ones: a backward pass that is right for a layer alone and wrong in a network, a test instance whose sizes hide the bug, an optimizer that applies its decay in the wrong place, a checkpoint that forgets one piece of state, a schedule measured in the wrong unit, an evaluation that quietly leaves dropout on.

Each milestone describes what to build (in the book's pseudocode, which fixes the interface and nothing else), explains what later work leans on it, gives a check to run before moving on, and lists the mistakes readers actually make. The **acceptance tests** at the end give inputs and the outputs a correct library produces, written so that any language can be tested against them. Every expected value was computed by a second, independent implementation, `scripts/book/worked/project_nnlib.py`, and is regenerated whenever the book is built. Use only your language's standard library plus the library of Volume I (the generator, the strided arrays, matrix multiplication, log-sum-exp and the numerical gradient).

![Seven boxes labelled with the milestones. Milestones 1 (layers), 3 (optimizers) and 4 (initialisation) sit in the top row. Milestone 2 (gradient checker) is below milestone 1, and milestone 5 (training loop) is below milestone 3 and receives arrows from milestones 1, 3 and 4. Milestone 6 (two models) receives arrows from milestones 5 and 2. Milestone 7 (debugging), highlighted, sits at the bottom and receives an arrow from milestone 6.](img/nnlib-milestones.svg)

The figure shows which milestone feeds which, so the order 1 to 7 is a valid build order. Build and test each milestone before starting the next. A failure in milestone 6 that traces back to milestone 1 costs an afternoon; the same failure caught by milestone 1's own check costs a minute.

Before the first line of code, six questions reach across the whole volume, and the last one reaches back to Volume I. Answer each from memory before reading on; the answers are at the end.

1. (Chapters 4 and 5, [Backpropagation Derived from Scratch](04-backpropagation.md) and [The Backward Pass of Every Layer](05-layer-gradients.md)) A linear layer computes $y_{nj} = \sum_i x_{ni} W_{ji} + b_j$ for a batch of $B$ rows. Given the gradient $dy$ of a scalar loss with respect to $y$, what are the gradients with respect to $W$, $b$ and $x$? What goes wrong if the backward pass assigns to $dW$ instead of adding to it?
2. (Chapter 6, [Proving Gradients Correct](06-gradient-checking.md)) A finite-difference check is run on a layer whose batch size, input width and output width are all 4. Name two bugs it can miss for that reason, and say how the sizes should be chosen.
3. (Chapter 7, [Optimizers: From SGD to AdamW](07-optimizers.md)) A weight $\theta = 1$ has a first gradient of $0.002$, the rate is $\eta = 0.1$ and the decay is $\lambda = 0.1$. By how much does AdamW change it in step 1, and what does it do first, before the Adam move? How does the coupled version differ?
4. (Chapter 8, [Learning-Rate Schedules and Gradient Clipping](08-schedules-and-clipping.md)) A schedule has warmup 10, peak $0.01$ and a cosine decay to a floor of $10^{-4}$ over 100 steps. What is the rate at step 0 and at step 9? A gradient of two tensors $(3, 4)$ and $(0, 12)$ is clipped to a global norm of 5: what are the new tensors?
5. (Chapter 9, [Initialization: Where Training Starts](09-initialization.md)) Why do the weights of a hidden layer with 48 inputs get variance $2/48$ in the He scheme, and why is the starting read-out of a classifier made smaller than the hidden layers?
6. (Chapter 11, [Regularization in Neural Networks](11-regularization.md), and Volume I, Chapter 17, [Randomness on a Deterministic Machine](../v01-mathematics/17-randomness.md)) A dropout layer processes a $[32, 32]$ batch from the book's SplitMix64 generator. How many numbers does it draw, in what order, and when is a unit kept? A model with dropout is evaluated twice on the same rows. When are the two answers equal?

## Milestones

A note on language and style. The pseudocode follows [Pseudocode Conventions](../pseudocode.md): 0-indexed, row-major, explicit broadcasts, explicit randomness, `require` for every input assumption. Use 64-bit floats throughout; the acceptance tests are computed in double precision. Store every tensor row-major, so that a model's parameters are one flat vector once concatenated and the optimizer needs no knowledge of layers. Write the checks of each milestone as a test file that you rerun after every change, and compare tables to the digits printed (a difference below half a unit in the last printed place), unless a milestone states another tolerance.

Almost every table below comes from a **fixture**, a small input defined by a formula instead of by random draws, so that you can reproduce it before your generator is right. The one rule is

$$
\mathrm{fix}(n;\, f, \phi, s)_k = s \,\sin(f\,k + \phi), \qquad k = 0, \dots, n - 1,
$$

with a frequency $f$, a phase $\phi$ and a scale $s$; a matrix is the fixture read row-major. Randomness appears only where the thing tested is random (dropout, initialization, the data, the shuffles), always from the book's SplitMix64 at a printed seed.

### Milestone 1: layers

The first milestone is ten layer types and two losses behind one contract. A layer owns its parameters, one gradient accumulator per parameter, and whatever its forward pass saves. In the book's pseudocode, with `T` standing for the layer type:

```text
record Linear:                                     # y = x W^T + b
    W: array<real>[n_out, n_in]
    b: array<real>[n_out]
    dW: array<real>[n_out, n_in]                   # accumulators, zeroed by zero_grad
    db: array<real>[n_out]
    x: array<real>[B, n_in]                        # saved by forward

function T_forward(layer: inout T, x: array<real>[B, n_in], mode: text,
                   rng: inout Generator) -> array<real>[B, n_out]
    # mode is "train" or "eval"; only dropout looks at it or at rng
function T_backward(layer: inout T, dy: array<real>[B, n_out]) -> array<real>[B, n_in]
    # adds to the accumulators, returns the gradient of the layer's input
function zero_grad(layer: inout T) -> none
```

The linear layer is the model for the rest. Its forward pass saves the input; its backward pass is the three rules of chapter 5, with the accumulation of its last section:

```text
function linear_backward(layer: inout Linear, dy: array<real>[B, n_out]) -> array<real>[B, n_in]
    dx = zeros(real, [B, n_in])
    for j in 0 .. n_out:
        for i in 0 .. n_in:
            for n in 0 .. B:
                layer.dW[j, i] = layer.dW[j, i] + dy[n, j] * layer.x[n, i]
        for n in 0 .. B:
            layer.db[j] = layer.db[j] + dy[n, j]
    for n in 0 .. B:
        for i in 0 .. n_in:
            for j in 0 .. n_out:
                dx[n, i] = dx[n, i] + dy[n, j] * layer.W[j, i]
    return dx
```

Because the loss is a mean over the batch, no layer divides by $B$; the division lives in the loss, once. Because the accumulators are added to, a loop must call `zero_grad` before each batch, and a batch may be cut into pieces whose backward calls add up to the full-batch gradient (chapter 5, "Many small batches, one big one").

The other layers, each with the formula that its backward pass must implement:

| layer | forward | backward (gradient of the input; gradients of the parameters) |
|---|---|---|
| tanh | $\tanh(z)$ | $dy\,(1 - \tanh^2 z)$ |
| ReLU | $\max(0, z)$ | $dy$ where $z > 0$, else $0$ (slope $0$ at $z = 0$) |
| GELU | $z\,\Phi(z)$, exact, with $\Phi(z) = \frac12(1 + \mathrm{erf}(z/\sqrt2))$ | $dy\,\bigl(\Phi(z) + z\,\varphi(z)\bigr)$, $\varphi$ the standard normal density |
| SiLU | $z\,\sigma(z)$ | $dy\,\sigma(z)\,(1 + z\,(1 - \sigma(z)))$ |
| layer norm | per row: $\hat x_d = (x_d - \mu)/\sigma_\epsilon$, $y_d = \gamma_d \hat x_d + \beta_d$, with $\sigma_\epsilon = \sqrt{\sigma^2 + \epsilon_{\mathrm{norm}}}$ and $\epsilon_{\mathrm{norm}} = 10^{-5}$ | $d\gamma_d \mathrel{+}= \sum_n dy_{nd}\hat x_{nd}$, $d\beta_d \mathrel{+}= \sum_n dy_{nd}$, and with $d\hat x = dy\,\gamma$: $dx_d = \frac{1}{\sigma_\epsilon}\bigl(d\hat x_d - \mathrm{mean}(d\hat x) - \hat x_d\,\mathrm{mean}(d\hat x\odot\hat x)\bigr)$, the means taken over the $D$ entries of the row |
| RMS norm | per row: $\hat x_d = x_d/\sqrt{\mathrm{mean}(x^2) + \epsilon_{\mathrm{norm}}}$, $y_d = \gamma_d \hat x_d$ | as layer norm without the term $\mathrm{mean}(d\hat x)$ and without $\beta$ |
| dropout | train: $y = a\,M$ with $M_{nj} = 1/(1 - p_{\mathrm{drop}})$ if the draw is at least $p_{\mathrm{drop}}$, else $0$; eval: $y = a$ | $da = dy\,M$ with the saved mask; the identity after an eval forward |
| residual | $y = x + f(x)$, $f$ a sequential block of the same input and output width | $dx = dy + f_{\mathrm{backward}}(dy)$ |
| sequential | the layers in order | the layers in reverse order |

Dropout draws one `rng.uniform()` per element of its input in row-major order (so $B \cdot n$ draws per forward call in train mode, none in eval mode); chapter 11 gives the convention and the reason the same mask must serve the backward pass. The two losses are not layers because they take a target, but they follow the same contract:

```text
function softmax_ce_forward(z: array<real>[B, C], label: array<int>[B]) -> real
    # mean over the batch of  logsumexp(z[n, :]) - z[n, label[n]],  the row maximum subtracted
    # saves p = softmax(z) row by row
function softmax_ce_backward() -> array<real>[B, C]
    # (p - onehot(label)) / B
function mse_forward(y: array<real>[B, O], t: array<real>[B, O]) -> real     # mean of (y - t)^2 over B * O entries
function mse_backward() -> array<real>[B, O]                                  # 2 (y - t) / (B * O)
```

Fusing softmax with cross-entropy (chapter 3) is what makes the gradient a subtraction and the loss finite for logits in the hundreds.

Everything after this milestone calls these twelve things and nothing else. A layer whose backward pass is right alone but whose state is clobbered when it is used inside a residual block, or twice in one model, fails only in milestone 6.

To check it, run each layer on the fixture of the table below and compare. Every layer sees the same input $x$ of shape $[3, 5]$ and a fixed probe $R$ in place of the arriving gradient, so that its scalar loss is $L = \sum_{n,j} y_{nj} R_{nj}$ and $dy = R$. The sizes 3, 5 and 7 are distinct on purpose; milestone 2 explains why.

<!-- worked: project_nnlib.layer_values -->
Input x: [3, 5], x[b, i] = sin(0.7 k + 0.3) with k = 5 b + i (smallest |x| = 0.0248, so ReLU is 2478 steps of the checker from its kink). Probe R: the same rule with frequency 1.1 and phase 0.4, shape [3, out]; the layer's scalar loss is L = sum(y * R), so dy = R.
linear: W[j, i] = 0.5 sin(0.9 k + 0.2) with k = 5 j + i, b[j] = 0.3 sin(1.3 j + 0.1). norms: gamma[i] = 1 + 0.3 sin(1.7 i + 0.5), beta[i] = 0.2 sin(2.3 i + 0.9), eps = 1e-5. dropout: p = 0.4 with Generator seed 0x6dde3aa86b97c782 (7916829689997936514), 15 uniform() draws in row-major order, kept when the draw is at least p (mask times 1/0.6: 0 0 0 1 1 1 1 1 1 1 0 0 1 0 1). residual: body = linear 5 -> 7 (the linear above), gelu, linear 7 -> 5 with W[j, i] = 0.5 sin(0.6 k + 0.7), b[j] = 0.2 sin(1.9 j + 0.3).

| layer | y shape | L | y[0,0] | y[2,last] | dx[0,0] | dx[2,4] |
|---|---:|---:|---:|---:|---:|---:|
| linear 5 -> 7 | 3 x 7 | +0.50268275 | +0.95320845 | +0.70757143 | +0.06920141 | -0.10117235 |
| tanh | 3 x 5 | -0.56927576 | +0.28720766 | -0.55464864 | +0.35729591 | -0.06363308 |
| relu | 3 x 5 | +0.57028160 | +0.29552021 | +0.00000000 | +0.38941834 | -0.00000000 |
| gelu | 3 x 5 | +0.26670191 | +0.18210007 | -0.16624526 | +0.28390933 | -0.00559231 |
| silu | 3 x 5 | +0.06020918 | +0.16943563 | -0.21791781 | +0.25142299 | -0.01899565 |
| layer norm | 3 x 5 | -1.11966819 | -0.73411084 | -2.22324210 | +2.29683056 | -0.67424360 |
| RMS norm | 3 x 5 | -1.07143569 | +0.50537748 | -1.10226874 | +0.42137769 | -0.31276818 |
| dropout p = 0.4 | 3 x 5 | -2.27512926 | +0.00000000 | -1.04178441 | +0.00000000 | -0.15317808 |
| residual (5 -> 7 -> 5) | 3 x 5 | -1.98236398 | +0.73346613 | -0.50463217 | +0.44365271 | +0.08888893 |

Parameter gradients after one backward pass from zero (tensor labels are k.name, k the index of the parameter-owning layer inside the layer under test):

| layer | tensor | g[0] | g[last] | sum of squares |
|---|---:|---:|---:|---:|
| linear 5 -> 7 | W | -0.55651470 | +0.53930345 | 8.69355106 |
| linear 5 -> 7 | b | +1.26740130 | +1.10517775 | 6.63008880 |
| layer norm | gamma | -0.93646570 | +0.28580449 | 1.36866158 |
| layer norm | beta | -0.90378685 | -1.85575727 | 12.57015175 |
| RMS norm | gamma | -0.59252728 | -0.33027344 | 1.00303327 |
| residual (5 -> 7 -> 5) | 0.W | +0.20429265 | +0.01014806 | 2.45017445 |
| residual (5 -> 7 -> 5) | 0.b | +0.23501051 | +0.13640733 | 1.08421958 |
| residual (5 -> 7 -> 5) | 2.W | -0.45737170 | -0.13688490 | 7.89364331 |
| residual (5 -> 7 -> 5) | 2.b | -0.90378685 | -1.85575727 | 12.57015175 |
<!-- /worked -->

The losses and the cases that break naive implementations:

<!-- worked: project_nnlib.loss_values -->
Logits z (softmax cross-entropy): z[b, j] = 2 sin(0.8 k + 0.5), k = 4 b + j, shape [3, 4]; labels (2, 0, 3). Mean squared error: the outputs y are the same 12 numbers as z (y = z, shape [3, 4]) and the targets are t[b, j] = 1.5 sin(1.4 k + 0.2), the mean taken over all 12 entries.

| loss | value | d[0,0] | d[0,2] | d[2,3] | sum of d below 1e-15 |
|---|---:|---:|---:|---:|---:|
| softmax cross-entropy | +1.67574842 | +0.05203084 | -0.22123189 | -0.30760407 | True |
| mean squared error | +2.79419961 | +0.11014118 | +0.25245645 | +0.01454639 | - |

Overflow case, one row z = (710, 0, -5): label 0 gives loss 0.00000000, label 1 gives loss 710.00000000; every value is finite and the gradient row sums to zero (below 1e-15: True).
<!-- /worked -->

The loss for a row of logits near 710 is finite only because the maximum is subtracted before the exponential. The last check of the milestone is the one the linear layer's accumulators exist for:

<!-- worked: project_nnlib.accumulation -->
Linear 5 -> 7 of the layer fixtures, x of shape [3, 5], dy of shape [3, 7] (the probe). 'once' is one backward call after zero_grad. The last row cuts the batch into rows {0} and {1, 2} and calls forward and backward on each with the matching rows of dy. A difference at the level of one rounding error (under 1e-15) is printed as a bound: its value depends on the order in which the sums are formed.

| experiment | largest |difference| from the stated value | expected |
|---|---:|---|
| backward twice without zero_grad | below 1e-15 | twice the gradient |
| zero_grad, then backward again | below 1e-15 | the gradient |
| rows 0 and rows 1 to 2 as two backward calls | below 1e-15 | the gradient |

The 'once' gradient has first entry -0.55651470 (the weight W[0,0]) and last entry +1.10517775 (the bias b[6]).
<!-- /worked -->

Six slips recur at this stage.

- Saving the output instead of the input (or the reverse) where the backward pass needs the other. The activations need the input $z$; tanh may use either, but only one is right for GELU.
- Computing the layer-norm gradient of the input with the formula for RMS norm, or the reverse. The two differ by one term, and milestone 2 shows what that looks like.
- Assigning `dW = ...` instead of `dW = dW + ...`. One layer in isolation passes; a layer used twice in one model, a batch cut into pieces and a residual block share-and-sum all fail.
- Dividing by $B$ in a layer as well as in the loss.
- Letting dropout draw in eval mode, or draw its mask a second time in the backward pass (chapter 11).
- Writing `if mode == "train"` for dropout but forgetting that the residual and sequential wrappers must pass `mode` and `rng` on.

### Milestone 2: the gradient checker

The second milestone is the checker of chapter 6, in two forms, and a policy for the instance it runs on.

```text
function within_gate(a: real, n: real, atol: real, rtol: real) -> bool
    return abs(a - n) <= atol + rtol * max(abs(a), abs(n))

function check_entries(loss: function(array<real>[P]) -> real, theta: array<real>[P],
                       grad: array<real>[P], indices: list<int>, h: real,
                       atol: real, rtol: real) -> list<int>        # the failing indices

function check_directional(loss: function(array<real>[N]) -> real, grad: array<real>[N],
                           w: array<real>[N], h: real, n_dirs: int, atol: real, rtol: real,
                           rng: inout Generator) -> bool           # best of n_dirs +-1 directions

function distinct_dims(dims: list<int>) -> bool
    return all(d > 1 for d in dims) and no two entries of dims are equal
```

`check_entries` and `check_directional` are the algorithms of chapter 6 unchanged. The project's gate for double precision is $h = 10^{-5}$, $\mathrm{atol} = 10^{-7}$, $\mathrm{rtol} = 10^{-5}$. The new part is a function that applies them to a **layer** in the standard way: the layer's scalar loss is $L = \sum y\,R$ for the fixed probe $R$, and the vector that is perturbed is the layer's input and every parameter tensor together. Checking the input as well as the parameters is not optional. A fault in the gradient of the input cannot be seen in the layer's own parameters; it corrupts the parameters of every layer below it.

```text
function check_layer(layer: inout T, x: array<real>[B, n_in], R: array<real>[B, n_out],
                     mode: text, dropout_seed: optional u64) -> list<(text, int, int, real, bool)>
    # For the input and for each parameter tensor, in order: (name, entries, failing, worst ratio, directional verdict).
    # loss(theta) copies theta into the input and parameters, restarts Generator(dropout_seed)
    # if there is one (the mask is then the same on every evaluation), runs forward, returns sum(y * R).
    # grad is the layer's backward(R) from zero accumulators; the worst ratio is the largest of
    # abs(a - n) / (atol + rtol * max(abs(a), abs(n))).
```

Restarting the generator before every evaluation is how dropout is checked: the function being differentiated is the one with the mask held fixed, and the backward pass must use the same mask (chapter 11).

The checker is the only thing that tells you the backward pass of a layer is the derivative of its forward pass. The loop of milestone 5 would train a wrong gradient without complaint, and chapter 12 showed how many ways a loss curve can look plausible while it does.

The check is that every layer of milestone 1 passes: no failing entries, and every worst ratio far below 1.

<!-- worked: project_nnlib.gradient_check_layers -->
Fixtures of the previous table. Per-entry check: central difference with h = 1e-5 on every entry of the input and of every parameter tensor, gate |a - n| <= 1e-7 + 1e-5 max(|a|, |n|); 'worst' is the largest of |a - n| / (1e-7 + 1e-5 max(|a|, |n|)), which must stay below 1. Directional check: best of 4 random +-1 directions, h = 1e-5, directions from Generator seed 0xdd1455d8ff629511 (15930452172402234641) (one integer(2) per entry, tensors in the order of the table, 4 directions each; +1 when the draw is 1). The two losses are checked per entry only. The dropout rows hold the mask fixed by restarting the same generator before every evaluation.

| layer | tensor | entries | failing | worst ratio | directional |
|---|---:|---:|---:|---:|---:|
| linear 5 -> 7 | input | 15 | 0 | 4.55e-06 | pass |
| linear 5 -> 7 | W | 35 | 0 | 2.59e-05 | pass |
| linear 5 -> 7 | b | 7 | 0 | 3.39e-06 | pass |
| tanh | input | 15 | 0 | 5.33e-06 | pass |
| relu | input | 15 | 0 | 9.88e-07 | pass |
| gelu | input | 15 | 0 | 2.28e-05 | pass |
| silu | input | 15 | 0 | 5.70e-06 | pass |
| layer norm | input | 15 | 0 | 7.95e-05 | pass |
| layer norm | gamma | 5 | 0 | 1.16e-05 | pass |
| layer norm | beta | 5 | 0 | 8.75e-07 | pass |
| RMS norm | input | 15 | 0 | 2.11e-05 | pass |
| RMS norm | gamma | 5 | 0 | 4.58e-06 | pass |
| dropout p = 0.4 | input | 15 | 0 | 4.13e-06 | pass |
| residual (5 -> 7 -> 5) | input | 15 | 0 | 3.88e-06 | pass |
| residual (5 -> 7 -> 5) | W | 35 | 0 | 6.28e-05 | pass |
| residual (5 -> 7 -> 5) | b | 7 | 0 | 2.18e-05 | pass |
| residual (5 -> 7 -> 5) | W | 35 | 0 | 7.49e-05 | pass |
| residual (5 -> 7 -> 5) | b | 5 | 0 | 1.67e-06 | pass |
| softmax cross-entropy | logits | 12 | 0 | 3.99e-05 | - |
| mean squared error | outputs | 12 | 0 | 5.88e-05 | - |
<!-- /worked -->

Then plant a fault. A common one is to reuse the RMS-norm backward for layer norm, which is exactly the formula of the table with one term dropped. The checker must reject it, and it must reject it in the input:

<!-- worked: project_nnlib.broken_layer -->
The layer-norm fixture of the previous tables. The planted fault is the backward pass of RMS norm used for layer norm: dx = inv_std (dxhat - xhat mean(dxhat xhat)), the term mean(dxhat) is missing. Directions: Generator seed 0x69c49541e226e675 (7621380579617334901), 4 per tensor, in the order of the table, correct layer first.

| backward pass | tensor | entries | failing | worst ratio | directional |
|---|---:|---:|---:|---:|---:|
| correct | input | 15 | 0 | 7.95e-05 | pass |
| correct | gamma | 5 | 0 | 1.16e-05 | pass |
| correct | beta | 5 | 0 | 8.75e-07 | pass |
| without the centring term | input | 15 | 15 | 6.51e+04 | FAIL |
| without the centring term | gamma | 5 | 0 | 1.16e-05 | pass |
| without the centring term | beta | 5 | 0 | 8.75e-07 | pass |
<!-- /worked -->

The parameter gradients of the faulty layer pass, because they do not use the faulty term; only the input gradient fails, on 15 of 15 entries with a worst ratio above $10^4$. A check that perturbed the parameters alone would have declared the layer correct.

The last check is the test-dimension policy. Take the fault "the gradient of the mean squared error divides by $B$ where $B \cdot O$ belongs", and run the same check at four instance sizes:

<!-- worked: project_nnlib.dims_policy -->
Fault: the gradient of the mean squared error divides by B where B * O belongs. Model: linear D -> O on a batch of B, fixtures as in the layer table (weights 0.5 sin(0.9 k + 0.2), bias 0.3 sin(1.3 j + 0.1), inputs sin(0.7 k + 0.3), targets 1.5 sin(1.4 k + 0.2)); per-entry check on every parameter, same gate. 'Distinct' is the policy test: every size above 1, no two equal.

| (B, D, O) | distinct | failing entries, correct backward | failing entries, with the fault |
|---|---:|---:|---:|
| (1, 1, 1) | no | 0 of 2 | 0 of 2 |
| (3, 5, 1) | no | 0 of 6 | 0 of 6 |
| (3, 5, 3) | no | 0 of 18 | 18 of 18 |
| (3, 5, 2) | yes | 0 of 12 | 12 of 12 |
<!-- /worked -->

At $(1, 1, 1)$ and at $O = 1$ the fault computes the right gradient, because $B\cdot O = B$; the policy `distinct_dims` says "no" for both. Equal sizes such as $(3, 5, 3)$ happen to expose this fault but not every fault, and the policy rejects them too: it is a rule about what an instance can reveal, not a prediction about one bug. Only a set of sizes that passes the policy, here $(3, 5, 2)$, is guaranteed to separate the roles of the three sizes.

Finally the models of milestone 6 are checked as wholes, once their layers pass. A tensor is labelled `k.name`, with `k` the 0-based position of a leaf layer (a layer that is not a container) in the depth-first order of the model's leaves, a residual wrapper contributing the leaves of its body in place, and `name` one of `W`, `b`, `gamma`, `beta`; the flat parameter vector of a model lists the tensors in that order, `W` before `b` (each row-major) and `gamma` before `beta`. The large tensors are checked directionally, the small ones (biases, gains, shifts) per entry, because those are the tensors that a backward pass folds and sums over a batch (chapter 6).

<!-- worked: project_nnlib.model_gradient_checks -->
Classifier of milestone 6 (hidden width 32, dropout p = 0.3 active, mask held fixed by restarting Generator seed 0x05f61e5542677ff4 (429564165995593716) before every evaluation) on training rows 0 to 4: loss 1.046265, 1507 parameters. Tensors of at most 64 entries are checked per entry (gate ratio below 1 means pass), the larger ones by the directional check with 4 directions from Generator seed 0xc983d605c60ecc45 (14520684943858650181).

| tensor | entries | check | failing | worst ratio / verdict |
|---|---:|---:|---:|---:|
| 0.W | 256 | directional | - | pass |
| 0.b | 32 | per entry | 0 | 1.46e-04 |
| 2.gamma | 32 | per entry | 0 | 1.57e-04 |
| 2.beta | 32 | per entry | 0 | 1.43e-04 |
| 3.W | 1024 | directional | - | pass |
| 3.b | 32 | per entry | 0 | 1.61e-04 |
| 6.W | 96 | directional | - | pass |
| 6.b | 3 | per entry | 0 | 5.07e-06 |

Autoencoder of milestone 6 on training readings 0 to 7: loss 0.384792, 499 parameters, every entry checked per entry; failing entries in total: 0; the largest ratio of any tensor: 3.85e-05.
<!-- /worked -->

Several slips either hide a wrong gradient from the checker or fail a correct one.

- Checking parameters only. See the broken layer above.
- Testing at sizes that coincide (chapter 6, "Choosing test dimensions"), or with a batch of one.
- Letting the loss closure draw a fresh dropout mask on every evaluation. The check then compares a derivative with a difference of two different functions and fails everywhere.
- Using a ReLU input too close to zero. The fixture's smallest $|x|$ is far from the checker's step; if you change the fixture, test that margin (chapter 6, "Kinks").
- Tolerances copied from a 32-bit check. The gate here is for 64-bit arithmetic; a gate loose enough for 32 bits forgives wrong derivatives (chapter 6, "What a check proves").
- Perturbing a copy of the parameters and forgetting to restore the originals, so that every later test runs on perturbed weights.

### Milestone 3: optimizers, schedule and clip

Milestone 3 collects the update rules of chapter 7 and the schedule and the clip of chapter 8 as functions on flat arrays that keep their state in arrays the caller owns:

```text
function sgd_step(theta: inout array<real>[P], g: array<real>[P], eta: real) -> none
function momentum_step(theta: inout array<real>[P], g: array<real>[P], m: inout array<real>[P],
                       eta: real, beta: real, nesterov: bool) -> none
function adam_step(theta: inout array<real>[P], g: array<real>[P], m: inout array<real>[P],
                   v: inout array<real>[P], t: int, eta: real, beta1: real, beta2: real,
                   eps: real, lam: real, decoupled: bool) -> none      # t >= 1: the 1-based step
function lion_step(theta: inout array<real>[P], g: array<real>[P], m: inout array<real>[P],
                   eta: real, beta1: real, beta2: real, lam: real) -> none
function schedule_rate(t: int, peak: real, floor: real, warmup: int, hold: int, total: int) -> real
function global_norm_clip(grads: list<array<real>>, tau: real) -> real  # scales in place, returns the norm before
```

The bodies are the pseudocode of chapters 7 and 8. The conventions that the acceptance tests depend on, and that are easy to get one step wrong:

- `adam_step` takes the **1-based** step in its bias corrections $1 - \beta^t$; `schedule_rate` takes the **0-based** step. A loop passes `t + 1` to the first and `t` to the second.
- Decoupled decay is applied to the weight first, `theta = theta - eta * lam * theta`, and then the Adam move uses the moments computed from the unmodified gradient. Coupled decay (`decoupled = false`) adds `lam * theta` to the gradient before the moments.
- Lion's sign of an exactly zero interpolation is $0$, not $+1$ (chapter 7), and Lion's state is one array.
- The clip scale is `min(1, tau / (norm + 0.000001))` and the clip returns the norm **before** scaling, which is what the log records.
- The schedule at the last warmup step is exactly the peak (`peak * (t + 1) / warmup` at `t = warmup - 1`), and the cosine reaches the floor only at step `total` itself, one past the last step: at the last step `t = total - 1` the rate is still slightly above it (the table below prints the rate at `t = 99` of a run of 100 steps). A `total` that counts epochs instead of steps reaches the floor after that many steps, long before the run ends, which milestone 7 plants.

The loop of milestone 5 is these functions in a fixed order: clip, then schedule, then update. The order is part of the contract (chapter 8, "Clipping and Adam").

The check is the optimizer trajectories on a fixed gradient sequence, so that the update rule alone is tested and no layer is involved:

<!-- worked: project_nnlib.optimizer_trajectories -->
Start theta = (1.0, -2.0, 0.5). The gradient at step t (t = 0, 1, 2, ...) is g_t[i] = 0.5 sin(0.8 (3 t + i) + 0.3), independent of theta: g_0 = (+0.147760, +0.445604, +0.473150), g_1 = (+0.213690, -0.175392, -0.458083), g_2 = (-0.462907, -0.186938, +0.202425). Rate 0.1 for every rule and every step; Adam and AdamW use (0.9, 0.999, 1e-8), Lion (0.9, 0.99); t is the 1-based step in the bias correction. Rows give theta after the stated number of steps.

| rule | steps | theta[0] | theta[1] | theta[2] |
|---|---:|---:|---:|---:|
| SGD | 1 | +0.98522399 | -2.04456037 | +0.45268500 |
| SGD | 2 | +0.96385500 | -2.02702121 | +0.49849329 |
| SGD | 3 | +1.01014573 | -2.00832737 | +0.47825080 |
| SGD | 10 | +0.97137191 | -2.01779301 | +0.50383507 |
| momentum, beta 0.9 | 1 | +0.98522399 | -2.04456037 | +0.45268500 |
| momentum, beta 0.9 | 2 | +0.95055659 | -2.06712554 | +0.45590979 |
| momentum, beta 0.9 | 3 | +0.96564666 | -2.06874036 | +0.43856961 |
| momentum, beta 0.9 | 10 | +0.87874728 | -2.17877038 | +0.37215168 |
| Nesterov, beta 0.9 | 1 | +0.97192558 | -2.08466470 | +0.41010149 |
| Nesterov, beta 0.9 | 2 | +0.91935592 | -2.08743419 | +0.45881210 |
| Nesterov, beta 0.9 | 3 | +0.97922772 | -2.07019370 | +0.42296344 |
| Nesterov, beta 0.9 | 10 | +0.86224446 | -2.17868635 | +0.38877158 |
| Adam | 1 | +0.90000001 | -2.10000000 | +0.40000000 |
| Adam | 2 | +0.80068751 | -2.13507954 | +0.40364474 |
| Adam | 3 | +0.81885017 | -2.13708783 | +0.38755589 |
| Adam | 10 | +0.76231288 | -2.20970197 | +0.34494952 |
| Adam, coupled L2 0.1 | 1 | +0.90000000 | -2.10000000 | +0.40000000 |
| Adam, coupled L2 0.1 | 2 | +0.79998435 | -2.07323509 | +0.39413655 |
| Adam, coupled L2 0.1 | 3 | +0.78935984 | -2.01582998 | +0.36816091 |
| Adam, coupled L2 0.1 | 10 | +0.58187194 | -1.72172577 | +0.25929314 |
| AdamW, decay 0.1 | 1 | +0.89000001 | -2.08000000 | +0.39500000 |
| AdamW, decay 0.1 | 2 | +0.78178751 | -2.09427954 | +0.39469474 |
| AdamW, decay 0.1 | 3 | +0.79213229 | -2.07534503 | +0.37465894 |
| AdamW, decay 0.1 | 10 | +0.68398993 | -2.00387431 | +0.30853497 |
| Lion, 0.9, 0.99, decay 0.1 | 1 | +0.89000000 | -2.08000000 | +0.39500000 |
| Lion, 0.9, 0.99, decay 0.1 | 2 | +0.78110000 | -1.95920000 | +0.49105000 |
| Lion, 0.9, 0.99, decay 0.1 | 3 | +0.87328900 | -1.83960800 | +0.38613950 |
| Lion, 0.9, 0.99, decay 0.1 | 10 | +0.71295311 | -1.80776338 | +0.46083854 |
<!-- /worked -->

Read the table the way chapter 7 taught. SGD's first step is $\eta g$; momentum's first step equals it and its later ones grow; Adam's first step is $\eta\,\mathrm{sign}(g)$ up to the $\epsilon$ in its denominator, whatever the gradient's size; the coupled variant differs from AdamW from step 2 on, because its decay has been divided by the gradient scale (chapter 7); Lion's steps have size exactly $\eta$ plus the decay. Then the schedule and the clip:

<!-- worked: project_nnlib.schedule_and_clip -->
Schedule: peak 0.01, floor 1e-4, warmup 10, total 100; the second column holds the peak for 10 steps after the warmup. t is the 0-based step. The last step of the run is t = 99 = total - 1; its rate is not yet the floor (the floor is reached at t = total = 100, where the rate is 0.00010000, and stays there).

| t | hold 0 | hold 10 |
|---|---:|---:|
| 0 | 0.00100000 | 0.00100000 |
| 4 | 0.00500000 | 0.00500000 |
| 9 | 0.01000000 | 0.01000000 |
| 10 | 0.01000000 | 0.01000000 |
| 11 | 0.00999698 | 0.01000000 |
| 20 | 0.00970148 | 0.01000000 |
| 55 | 0.00505000 | 0.00601570 |
| 99 | 0.00010302 | 0.00010382 |

Clip: two gradient tensors (3, 4) and (0, 12), global norm 13; scale = min(1, tau / (norm + 1e-6)).

| tau | norm before | scale | tensor 1 after | tensor 2 after |
|---|---:|---:|---:|---:|
| 5.0 | 13.00000000 | 0.38461536 | (1.153846, 1.538461) | (0.000000, 4.615384) |
| 20.0 | 13.00000000 | 1.00000000 | (3.000000, 4.000000) | (0.000000, 12.000000) |
<!-- /worked -->

Optimizer faults rarely crash anything, so the usual ones are worth listing.

- Off-by-one between the two step indices. If your Adam first step is not $\eta g/(|g| + \epsilon)$ in magnitude, suspect the bias correction.
- Applying decay after the Adam move, or in the gradient. The first is a different, smaller change; the second is the coupled variant.
- Clipping each tensor by its own norm. The clip is one scalar for the whole model (chapter 8).
- Clipping after the optimizer's moments have been updated, or clipping the parameters instead of the gradient.
- Keeping the optimizer's moments in the layers, so that a checkpoint has to collect them from many places. Keep them next to the flat parameter vector.

### Milestone 4: initialization

Milestone 4 is the initializer of chapter 9, with one addition used by the models below:

```text
function init_linear(n_out: int, n_in: int, scheme: text, shape: text, rng: inout Generator,
                     factor: real) -> array<real>[n_out, n_in]
    # scheme "he": variance 2 / n_in;  "xavier": 2 / (n_in + n_out);  "fan_in": 1 / n_in
    # shape "normal": sigma * rng.normal();  "uniform": sigma * sqrt(3) * (2 * rng.uniform() - 1);
    # "truncated": redraw rng.normal() until abs(z) <= 2, then sigma * z / sqrt(0.7737)
    # sigma = factor * sqrt(variance); entries are drawn row-major, one draw per entry
```

The `factor` is the read-out factor of chapter 9 ("The output layer"): a classifier's last layer is multiplied by $0.1$ so that its first loss is near $\ln C$ instead of above it. Biases start at zero, layer-norm gains at one and shifts at zero. A model draws all of its weights from **one** generator, layer by layer in forward order, and nothing else draws from that generator.

The first loss, the first gradient norm and with them the whole first epoch depend on the scale of every layer (chapter 9), and reproducibility depends on the draw order being written down.

To check it, draw a $128 \times 96$ matrix per scheme and shape from the named streams and compare the first draws and the standard deviation with the target:

<!-- worked: project_nnlib.initialization_statistics -->
A weight matrix with n_out = 128 and n_in = 96 (12288 entries, row-major), drawn from a fresh generator per row named init.<scheme>.<shape> exactly, with nothing before it, e.g. the string init.he.normal (the seed of a stream is the first 8 bytes, read as a big-endian unsigned integer, of the SHA-256 digest of its name; the seeds are printed below, so no derivation is needed; the draw is the one call of the chapter-9 rule per entry, the truncated shape redrawing normal() until |z| <= 2). 'target sd' is sqrt(2/n_in), sqrt(2/(n_in+n_out)) or sqrt(1/n_in). 'sd of the 12288' is the population standard deviation with divisor n = 12288 after subtracting the mean of the entries (not the divisor n - 1 of the book's sd).

| scheme | shape | first three weights | target sd | sd of the 12288 (divisor n) | largest |w| |
|---|---|---:|---:|---:|---:|
| he | normal | +0.010290, +0.310372, -0.154359 | 0.144338 | 0.145302 | 0.5985 |
| he | uniform | +0.121690, +0.188562, -0.005744 | 0.144338 | 0.144823 | 0.2499 |
| he | truncated | +0.086434, -0.057611, +0.156911 | 0.144338 | 0.144787 | 0.3282 |
| xavier | normal | +0.188860, +0.067068, -0.053125 | 0.094491 | 0.093375 | 0.3505 |
| fan_in | normal | +0.006299, +0.127228, -0.067008 | 0.102062 | 0.102331 | 0.4063 |

Seeds: init.he.normal seed 0x21a24a7e65ef6de9 (2423581456215272937); init.he.uniform seed 0xa07ae76b8f7af7d8 (11563809442381887448); init.he.truncated seed 0x659daaac4ab1c264 (7322196226138554980); init.xavier.normal seed 0x0511b799f1cef27b (365274916608471675); init.fan_in.normal seed 0x42b18e0884be3cbb (4805778444618972347)
<!-- /worked -->

The first three weights are a bit-exact test of your generator, the order of the draws and the formula. The standard deviation of 12288 samples agrees with its target to within about $1\%$ by chance, so a test against the target alone uses a $3\%$ margin; the table's own value is the test for a bit-exact generator.

Initialization changes the model while leaving the statistics right, which makes these slips easy to commit.

- Drawing a matrix column-major or filling the layers in another order; the statistics are right and the model is different.
- Using the fan-out where chapter 9's rule uses the fan-in (He and the fan-in scheme), which flips the scale of a rectangular layer.
- Letting the truncated shape draw a fixed count of normals; it redraws until one is accepted, so the number of draws is not $n_{\mathrm{out}}\,n_{\mathrm{in}}$.
- Initializing biases with the weight rule.

### Milestone 5: the training loop

Milestone 5 is the loop of chapter 13 with three additions: early stopping, a checkpoint that is enough to resume exactly, and a log that is the data structure the later tables read.

```text
record TrainState:
    theta: array<real>[P]            # the parameters
    m: array<real>[P]                # AdamW's moments
    v: array<real>[P]
    t: int                           # steps taken
    epoch: int                       # epochs finished
    order: array<int>[N]             # the current example order: the next shuffle starts from it
    rng_batches: u64                 # the states of the generators, as integers
    rng_dropout: u64
    best_theta: array<real>[P]       # the checkpoint of the lowest validation loss so far
    best_val: real
    best_epoch: int
    wait: int                        # consecutive epochs without an improvement
    log: list<array<real>[7]>        # epoch, training loss, validation loss, accuracy or 0, mean grad norm, clipped steps, last rate

record TrainConfig:
    batch: int                       # examples per step; divides N
    peak: real                       # the schedule: warmup = total // 10, floor = peak / 100, hold 0, total = epochs * (N // batch)
    epochs: int
    tau: real                        # global-norm clip threshold
    lam: real                        # decoupled decay of AdamW (beta1 0.9, beta2 0.999, eps 1e-8)
    patience: optional int           # none: never stop early

function fit(state: inout TrainState, until_epoch: int, model: Sequential, loss: Loss,
             X: array<real>[N, n_in], T: list<array<real>>, Xv: array<real>[Nv, n_in], Tv: list<array<real>>,
             cfg: TrainConfig) -> none
    # T and Tv hold one target per row: the class index as a one-element array for softmax_ce, the target row for mse
    # for each epoch up to until_epoch, unless stopped:
    #   rng = Generator(state.rng_batches); shuffle(order, rng); state.rng_batches = rng.state   (the dropout generator likewise)
    #   for each batch: zero_grad, forward in "train" mode,
    #   loss, backward, global_norm_clip, adam_step(t + 1, schedule_rate(t)), set the parameters
    #   evaluate in "eval" mode on the whole training and validation sets; append to the log
    #   if val < best_val: best_theta = theta, best_val = val, best_epoch = epoch, wait = 0
    #   else: wait = wait + 1; stop when wait >= patience
function save(state: TrainState) -> bytes     # every field above, floats written so that they read back exactly
function load(bytes) -> TrainState
```

Four rules make the run reproducible, and each has a test below.

1. **One generator per purpose.** The shuffles have their own, dropout has its own, and initialization its own; none shares a stream with another. Adding a dropout layer must not change the order of the batches.
2. **The draw order is part of the contract.** One in-place shuffle of the current order per epoch, then one dropout draw per element per training step; evaluation draws nothing.
3. **A checkpoint holds everything the next step reads.** That is not only the parameters: the optimizer's moments and step count, the current example order, the state of every generator, the best parameters so far and the patience counter all belong in it. Write floats in a form that reads back to the same bits (17 significant digits, or the bytes).
4. **The best epoch is chosen by strictly lower validation loss.** The state before training counts as epoch 0, so a run that never improves keeps its initial weights; an equal loss is not an improvement. The first epoch at which `wait` reaches the patience is the last one run.

A result you cannot repeat is a result you cannot debug, and a run you cannot resume wastes the hours it took. Every experiment in Volumes V to X is this loop.

The check starts with the early-stopping rule on its own, as a function of a list of validation losses:

<!-- worked: project_nnlib.early_stopping_rule -->
Validation losses by epoch, epoch 0 (before training) first: 1.20, 0.90, 0.70, 0.55, 0.50, 0.51, 0.49, 0.50, 0.52, 0.53, 0.54, 0.40. A loss is an improvement only when strictly below the best so far; training stops at the epoch where the number of consecutive non-improvements reaches the patience.

| patience | best epoch (checkpoint kept) | stops after epoch |
|---|---:|---:|
| 1 | 4 | 5 |
| 2 | 6 | 8 |
| 3 | 6 | 9 |
| 5 | 11 | never |

A run of equal losses, 1.0, 0.8, 0.6, 0.6, 0.6, 0.6, with patience 3 keeps epoch 2, the first of the equal values, and stops after epoch 5.
<!-- /worked -->

Then the exact-resume test. Train 12 epochs in one go. Train 6, save, build a new model and trainer from scratch, load, train 6 more. The two end in the same parameters and the same log, compared with `==`, not within a tolerance. The same program run twice gives the same result, which is the determinism test, and each piece of state left out of the checkpoint produces a run that is no longer the same:

<!-- worked: project_nnlib.resume_identity -->
Classifier of milestone 6 (dropout p = 0.3, initial weights from Generator seed 0x9551c68b36b463fb (10759599285985371131)) on the first 128 training rows (4 steps of 32 per epoch, the 256 validation rows for the log), 12 epochs, no early stopping; AdamW (0.9, 0.999, 1e-8) with decay 0.01, clip 1, and the schedule of milestone 3 with peak 3e-3, warmup 4 steps (a tenth of the 48 steps), floor 3e-5 and total 48 (hold 0). Streams: shuffles seed 0x6508599f48ae21a6 (7280167338298450342), dropout seed 0x6c197b6cca9a741c (7789392737699787804). The three damaged checkpoints are made from the checkpoint saved after epoch 6: the order replaced by the sorted order 0 .. 127, the dropout generator replaced by a generator at its seed (the start of its stream), the moments m and v replaced by zeros (the step count kept). 'Identical' means every parameter and every logged number equal with ==, not within a tolerance.

| run | identical to the first row | largest parameter difference | training loss, epoch 12 | validation loss, epoch 12 |
|---|---:|---:|---:|---:|
| 12 epochs without a stop | - | - | 0.775070 | 0.980595 |
| 6 epochs, save, load, 6 epochs | yes | 0.00e+00 | 0.775070 | 0.980595 |
| same, run twice in one process | yes | 0.00e+00 | 0.775070 | 0.980595 |
| checkpoint without the shuffle order (sorted instead) | no | 8.84e-03 | 0.775851 | 0.980406 |
| checkpoint without the dropout generator state | no | 8.12e-03 | 0.775156 | 0.980098 |
| checkpoint without the optimizer moments | no | 1.07e-02 | 0.777260 | 0.981730 |
<!-- /worked -->

Every deviation in the last three rows is small in loss (at most about $2\times10^{-3}$), which is why this bug survives: the run looks fine and is not the run you saved.

A resumed run goes wrong quietly, and these are the usual ways.

- Saving the parameters and forgetting the moments, the step count, the order or a generator. The table shows each.
- Reshuffling a freshly sorted order every epoch instead of shuffling the previous epoch's order in place. Both are valid shuffles and give different batches; pick the in-place rule and write it down.
- Calling the model in train mode for the evaluation. Milestone 7 shows what that does.
- Counting the patience in steps, or counting an equal loss as an improvement.
- Evaluating only the validation loss and logging nothing else. The log of the table in milestone 6 (norm of the gradient, number of clipped steps, rate) is what lets you debug a run after the fact.

### Milestone 6: two models trained end to end

Milestone 6 trains two models from the library, with the checks of chapter 12 before each run.

**The classifier.** The task is the synthetic three-class task of chapter 12, redrawn from this chapter's own stream so that all of its numbers are new. There are 8 inputs. Three pairs of teacher vectors $\mathbf{u}_k$ and $\mathbf{v}_k$ (8 numbers each) score the input, and the label is the class whose score $2\,(\mathbf{u}_k\cdot\mathbf{x}) + (\mathbf{v}_k\cdot\mathbf{x})^2$ is largest (the first index on a tie). A hidden layer is needed because of the squared term.

<!-- worked: project_nnlib.classifier_task -->
Generator seed 0x13c5662d85d14f3d (1424657202837278525); draws in order: u_0, u_1, u_2, v_0, v_1, v_2 (8 normal() each), then 1024 inputs (8 normal() each, row by row). Label = the first index of the largest of 2 (u_k . x) + (v_k . x)^2, k = 0, 1, 2. Rows 0 to 511 are the training set, 512 to 767 the validation set, 768 to 1023 the test set. u_0 starts (+0.547752, -1.690484, +2.335700); training row 0 starts (-1.359510, -1.196394, -0.232255) with label 0; training row 511 ends (-0.204668, -0.527013, +1.117383) with label 2; sum of the 4096 training inputs +21.719003.

| split | rows | class 0 | class 1 | class 2 |
|---|---:|---:|---:|---:|
| training | 512 | 149 | 131 | 232 |
| validation | 256 | 63 | 83 | 110 |
| test | 256 | 72 | 81 | 103 |
<!-- /worked -->

The model uses every kind of layer once: a linear layer 8 to 32 with GELU, a pre-norm residual block, and a linear read-out to 3 logits.

```text
function make_classifier(p_drop: real, rng: inout Generator) -> Sequential
    first = Linear(8 -> 32,  W = init_linear(32, 8,  "he",     "normal", rng, 1.0))
    inner = Linear(32 -> 32, W = init_linear(32, 32, "he",     "normal", rng, 1.0))
    last  = Linear(32 -> 3,  W = init_linear(3,  32, "fan_in", "normal", rng, 0.1))
    return Sequential([first, GELU, Residual(Sequential([LayerNorm(32), inner, GELU, Dropout(p_drop)])), last])
```

The weights are drawn in the order `first`, `inner`, `last`. The loss is `softmax_ce`. Before training, run the checklist of chapter 12 that applies: the loop of milestone 2 on the whole model, the initial loss against $\ln 3$, the label prior, a linear baseline, and the overfit-one-batch test on 8 training rows with dropout off. Then train. The full settings are in the head of the table.

<!-- worked: project_nnlib.classifier_training -->
Model: Linear 8 -> 32 (He), GELU, residual block [layer norm 32, Linear 32 -> 32 (He), GELU, dropout 0.3], Linear 32 -> 3 (1/n_in scale times 0.1); 1507 parameters, biases and layer-norm shifts 0, gains 1. Start draws: Generator seed 0x9551c68b36b463fb (10759599285985371131), weights of the three linear layers in order, row-major, one normal() each. Loop: batch 32 (16 steps per epoch, at most 40 epochs = 640 steps), shuffles of 0 .. 511 from Generator seed 0x211ccd342efd4ba8 (2386007526609865640) (one in-place shuffle per epoch), dropout draws from Generator seed 0x4f307fd441988ba6 (5706201277488728998) (one uniform() per element of the 32 x 32 block, row-major, per step). AdamW (0.9, 0.999, 1e-8), decay 0.01, clip 1.0, peak rate 0.003, warmup 64 steps, cosine to 3.0e-05, early stopping with patience 8 on the validation loss. Losses and accuracy are computed in eval mode over the whole set after each epoch; 'grad norm' is the mean norm before the clip, 'clipped' the number of clipped steps in the epoch, 'rate' the rate of the epoch's last step. Linear softmax baseline: one linear layer 8 -> 3 (weights N(0, 1/8) row-major from Generator seed 0xb55c5dfb7d7f3d96 (13068423553446002070), biases 0), trained on the same data by the same loop (batch 32, shuffles from the classifier's shuffle stream, no dropout, no early stopping) for 20 epochs of 16 steps with AdamW decay 0.01, clip 1 and peak rate 0.01 (warmup 32 steps, cosine to 1e-4); validation loss and accuracy in eval mode after epoch 20. Overfit one batch: the classifier above with dropout in eval mode (off), training rows 0 to 7, a constant rate 0.01 (no schedule), AdamW without clip and without decay, 150 steps, the loss before the first step and after the 150th and the accuracy on those 8 rows.

| before training | value |
|---|---:|
| ln 3, the loss of a model that ignores the input | 1.09861 |
| initial loss, training set (eval mode) | 1.12218 |
| training class counts; majority-class validation accuracy | [149, 131, 232]; 0.4297 |
| validation loss of the training prior | 1.08586 |
| linear softmax baseline (20 epochs), validation loss and accuracy | 0.92078, 0.5781 |
| overfit one batch (8 rows, 150 steps of AdamW 0.01, eval mode): loss and accuracy | 1.08257 -> 3.71e-06, 1.000 |

| epoch | training loss | validation loss | validation accuracy | grad norm | clipped | rate |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 1.12218 | 1.13774 | 0.2344 |  |  |  |
| 1 | 1.08317 | 1.09898 | 0.3906 | 1.6666 | 16 | 7.500e-04 |
| 2 | 1.01729 | 1.03344 | 0.4531 | 1.4600 | 16 | 1.500e-03 |
| 5 | 0.73949 | 0.76798 | 0.6758 | 1.1801 | 14 | 2.995e-03 |
| 10 | 0.30059 | 0.39248 | 0.8281 | 1.1226 | 10 | 2.805e-03 |
| 15 | 0.16935 | 0.29437 | 0.8789 | 0.9393 | 7 | 2.373e-03 |
| 19 | 0.12154 | 0.23893 | 0.8984 | 0.9356 | 7 | 1.907e-03 |
| 27 | 0.08804 | 0.24443 | 0.8984 | 0.8371 | 4 | 8.948e-04 |

Training stopped after epoch 27; the checkpoint kept is epoch 19, with validation loss 0.23893 and accuracy 0.8984. On the test set the checkpoint scores loss 0.23052 and accuracy 0.9141.
<!-- /worked -->

The initial loss is $1.12218$, close to $\ln 3 = 1.09861$ (the read-out factor of milestone 4 did that; at factor 1 the same model starts well above it). The linear baseline reaches $0.5781$ and the prior $0.4297$, so the task needs the hidden layer, and the network clears both. Training loss falls below validation loss and stays below it, the gap of a model that dropout and the stop keep from memorising the 512 rows. The stop is early stopping at work: the best epoch is 19, the validation loss rises a little after it, and the run ends eight epochs later with the epoch-19 checkpoint, whose test accuracy is $0.9141$.

**The autoencoder.** The data, the architecture and the settings are those of chapter 13; this milestone asks for the same model out of your library, using the streams of that chapter by name so that its table is a test of yours: `first_models.ae_data`, `first_models.ae_init`, `first_models.batches`. The layers are linear $16\to12$, GELU, linear $12\to3$, linear $3\to12$, GELU, linear $12\to16$, with weights from `init_linear(n_out, n_in, "fan_in", "normal", rng, 1.0)` and the loss `mse`.

<!-- worked: project_nnlib.autoencoder_training -->
Data, start and shuffles are the streams of chapter 13: `first_models.ae_data` (draws per reading: s1, s2, s3, then n_0 .. n_15; 512 readings split 256 / 128 / 128), `first_models.ae_init` (weights N(0, 1/n_in) of the four linear layers in order, row-major, biases 0) and `first_models.batches` (one shuffle of 0 .. 255 per epoch). 499 parameters. AdamW (0.9, 0.999, 1e-8), decay 0.01, clip 1, batch 32 (8 steps per epoch), 300 epochs = 2400 steps, peak rate 0.01, warmup 240 steps, cosine to 1e-4, no early stopping. The target of every reading is itself.

| check before training | value |
|---|---:|
| mean predictor, training and validation mean squared error | 0.6969, 0.5804 |
| initial loss, training and validation | 1.0337, 0.8172 |
| gradient check, all 499 entries on readings 0 to 7: entries failing | 0 |
| overfit one batch (8 readings, 400 steps of AdamW 0.01): loss | 0.3848 -> 1.82e-04 |

| epoch | training loss | validation loss | grad norm | clipped | rate |
|---|---:|---:|---:|---:|---:|
| 0 | 1.03367 | 0.81718 |  |  |  |
| 1 | 1.00545 | 0.79637 | 1.6061 | 7 | 3.33e-04 |
| 2 | 0.93328 | 0.74333 | 1.4314 | 6 | 6.67e-04 |
| 5 | 0.67616 | 0.56194 | 0.5516 | 0 | 1.67e-03 |
| 10 | 0.31445 | 0.27236 | 0.3902 | 0 | 3.33e-03 |
| 20 | 0.06187 | 0.06101 | 0.0739 | 0 | 6.67e-03 |
| 40 | 0.00862 | 0.00860 | 0.0727 | 0 | 9.97e-03 |
| 60 | 0.00393 | 0.00411 | 0.0872 | 0 | 9.70e-03 |
| 80 | 0.00306 | 0.00335 | 0.0603 | 0 | 9.19e-03 |
| 100 | 0.00271 | 0.00301 | 0.0432 | 0 | 8.45e-03 |
| 150 | 0.00239 | 0.00267 | 0.0296 | 0 | 5.92e-03 |
| 200 | 0.00226 | 0.00251 | 0.0254 | 0 | 3.10e-03 |
| 250 | 0.00220 | 0.00247 | 0.0199 | 0 | 9.18e-04 |
| 300 | 0.00219 | 0.00246 | 0.0164 | 0 | 1.00e-04 |

Best validation loss 0.00246 at epoch 281; the checkpoint kept is that epoch's, with test loss 0.00254. Mean predictor on the test readings: 0.72837.
<!-- /worked -->

The run reproduces the log of chapter 13 to the digits printed there: the loss at epoch 1 is $1.00545$, the best validation loss is at epoch 281, and the test loss of the checkpoint is $0.00254$ against $0.72837$ for the mean predictor. That agreement is an independent test of the library: that chapter's code shares nothing with this chapter's but the generator and the arithmetic it describes. (Your library computes in its own order of summation, so an agreement to five decimals is the expected result and a bit-exact one is not.)

These two are the smallest models that exercise every part: a classifier with a normalization, a residual connection and dropout, and a regression with a bottleneck. A library that trains both and passes every check is the base on which the engines of later volumes are compared.

The checks are the gradient checks above; the initial losses; the tables to the stated tolerance; and the two tests that make a result trustworthy and not merely good: the same program twice gives the same table, and the stopped and resumed run gives the same table (milestone 5).

Most failures in the two models come down to one of these.

- Drawing the classifier's weights in the wrong layer order, or giving the residual block's inner layer another generator.
- Evaluating in train mode (the loss and the accuracy then depend on the dropout draws).
- Computing the autoencoder's loss as a sum or as a mean over the batch only. The gradient of the mean over $B\cdot D$ entries is $2(\hat x - x)/(BD)$ (chapter 13).
- Reporting the final-epoch validation loss with the best-epoch checkpoint, or the validation loss of the best epoch as the estimate of the error on new data. The test set is scored once, at the end.
- A residual block whose branch output width differs from its input width; the wrapper must `require` equal widths.

### Milestone 7: finding three planted defects

Milestone 7 builds nothing new; it is a diagnosis. A colleague hands you a driver for the classifier of milestone 6. It uses your library and the settings below, and it trains a model that does not work. You are to find what is wrong, using the practice of chapter 12 in its order of cost, and to fix it. There are three defects. They are in the driver, not in your library (milestones 1 to 6 are done and checked), and they hide one another: the first one you find is not the one that is easiest to see.

```text
function colleague_driver() -> list<array<real>[7]>
    (X, y) = training_set()                                  # the 512 rows of the task, true labels
    (Xv, yv) = validation_set()
    y_in = [y[(n + 1) mod 512] for n in 0 .. 512]            # the labels paired with the inputs
    model = make_classifier(0.3, Generator(init_seed))
    S = 512 // 32                                            # steps per epoch
    epochs = 40
    total = epochs                                           # for the schedule
    warmup = total // 10
    ...                                                      # as the loop of milestone 5, with y_in as the targets,
                                                             # peak 0.003, floor 0.00003, decay 0.01, clip 1, no early stopping
    for epoch in 1 ..= epochs:
        ...                                                  # the training steps
        train_loss = loss_only(model, X, y_in, "eval")
        val_loss = loss_only(model, Xv, yv, "train", rng_dropout)
        append [epoch, train_loss, val_loss, ...] to the log
```

The streams are those of milestone 6. In the shipped driver the validation pass uses the same dropout generator as training, once before the first epoch and once after every epoch.

Read the table first as a symptom report, not as a hint. Each row is a run of the library with the defects removed one at a time, the first being the driver as shipped:

<!-- worked: project_nnlib.debug_stages -->
The classifier of milestone 6 with every other setting unchanged, early stopping off, 40 epochs of 16 steps. Same streams as milestone 6; in the shipped configuration the validation pass runs in train mode and so consumes the dropout generator (once before training, 8192 draws, and once after each epoch, in that order relative to the training steps). Loss columns are eval-mode losses for the training set and the configuration's own validation pass for the validation set; 'rate' is the rate of the epoch's last step.

| stage | epoch | training loss | validation loss | validation accuracy | rate |
|---|---:|---:|---:|---:|---:|
| 0: as shipped | 1 | 1.0596 | 1.0998 | 0.4336 | 2.367e-03 |
|  | 2 | 1.0527 | 1.1005 | 0.4336 | 4.649e-04 |
|  | 5 | 1.0507 | 1.0991 | 0.4297 | 3.000e-05 |
|  | 10 | 1.0493 | 1.0877 | 0.4336 | 3.000e-05 |
|  | 20 | 1.0470 | 1.0836 | 0.4414 | 3.000e-05 |
|  | 40 | 1.0439 | 1.0870 | 0.4375 | 3.000e-05 |
| 1: labels aligned | 1 | 0.9875 | 1.0137 | 0.4883 | 2.367e-03 |
|  | 2 | 0.9442 | 0.9726 | 0.5391 | 4.649e-04 |
|  | 5 | 0.9384 | 0.9564 | 0.5742 | 3.000e-05 |
|  | 10 | 0.9334 | 0.9578 | 0.5781 | 3.000e-05 |
|  | 20 | 0.9232 | 0.9443 | 0.5938 | 3.000e-05 |
|  | 40 | 0.9038 | 0.9280 | 0.6094 | 3.000e-05 |
| 2: and horizon in steps | 1 | 1.0830 | 1.1003 | 0.3867 | 7.500e-04 |
|  | 2 | 1.0177 | 1.0364 | 0.4648 | 1.500e-03 |
|  | 5 | 0.7306 | 0.7601 | 0.6562 | 2.995e-03 |
|  | 10 | 0.3085 | 0.4499 | 0.7891 | 2.805e-03 |
|  | 20 | 0.1153 | 0.3275 | 0.8594 | 1.781e-03 |
|  | 40 | 0.0775 | 0.2975 | 0.8984 | 3.002e-05 |
| 3: and eval mode (healthy) | 1 | 1.0832 | 1.0990 | 0.3906 | 7.500e-04 |
|  | 2 | 1.0173 | 1.0334 | 0.4531 | 1.500e-03 |
|  | 5 | 0.7395 | 0.7680 | 0.6758 | 2.995e-03 |
|  | 10 | 0.3006 | 0.3925 | 0.8281 | 2.805e-03 |
|  | 20 | 0.1149 | 0.2427 | 0.8984 | 1.781e-03 |
|  | 40 | 0.0784 | 0.2470 | 0.8906 | 3.002e-05 |
<!-- /worked -->

![Validation loss over 40 epochs of the four runs of the debugging exercise. The shipped run stays near 1.09, the level of a model that ignores the input; the run with the labels aligned falls slowly to 0.93; the run with the schedule also repaired falls to about 0.30 and the run with all three repaired to about 0.25.](img/nnlib-debug-stages.svg)

The shipped run's validation loss never leaves $\ln 3$ and its accuracy stays at $0.44$, close to the share of the majority class. That is one symptom with several causes (chapter 12, "Plateau"). The checklist separates them, and the measurements below are what each item reports on the shipped configuration and on the healthy one:

<!-- worked: project_nnlib.debug_diagnostics -->
Checklist measurements of the shipped configuration against the healthy one (stage 3 of the previous table). Validation labels are never shifted. The linear baseline and the overfit-one-batch test have the settings of the milestone-6 table, trained on the shifted labels (training label i replaced by the label of row i + 1, the last by the first) in the shipped column. The determinism probe evaluates the final weights of the run of its own column (shipped: stage 0, healthy: stage 3) twice on the 256 validation rows, in train mode with a fresh generator seed 0xd7ed388ec07ca909 (15559154473400838409) per column (the draws continue between the two calls) and then twice in eval mode.

| measurement | shipped | healthy | reading |
|---|---:|---:|---|
| initial loss against ln 3 = 1.09861 | 1.12299 | 1.12218 | same: not the fault |
| gradient check of the model on rows 0 to 4 (milestone 2): failing entries, directional verdicts | 0 failing, all pass | 0 failing, all pass | same: the gradient is of the loss it is given |
| linear baseline, validation accuracy (majority class 0.4297) | 0.4531 | 0.5781 | shipped: barely above the majority class |
| overfit one batch: final loss | 5.57e-06 | 3.71e-06 | both pass: memorising needs no signal |
| epoch-1 rate and rate at epoch 5 (peak 0.003, floor 3e-05) | 2.367e-03, 3.000e-05 | 7.500e-04, 2.995e-03 | shipped: the floor by epoch 5 |
| rate of the last step of epoch 40 | 3.000e-05 | 3.002e-05 | both at the floor 3e-05 to three digits; the shipped one got there 600 steps early |
| two evaluations of one model, train mode: loss difference | 0.0051 | 0.0007 | different answers for one input |
| two evaluations of one model, eval mode: loss difference | 0.0e+00 | 0.0e+00 | eval mode: identical, as it must be |
<!-- /worked -->

Go down the list as chapter 12 orders it.

1. **Shapes and the initial loss.** Both configurations start at $1.12$, near $\ln 3$. This item finds nothing, and its silence is information: the model, the loss and the initialization are not the cause.
2. **The gradient check of the whole model.** It passes for the shipped driver as for the healthy one. The gradient is right for the loss the driver gives it; the check cannot know the labels are paired wrongly.
3. **The label prior and the linear baseline.** The prior's accuracy is $0.4297$. A linear model trained by the same loop on the shipped labels reaches $0.4531$, barely above it, against $0.5781$ on the true labels. A model that does worse than a straight line fitted to the true labels, on the same inputs, is being taught the wrong thing. This is the first defect: **the labels do not belong to the inputs**. Fix it (stage 1).
4. **Overfit one batch.** It passes on the shipped labels (final loss $5.57\times10^{-6}$), exactly as on the true ones: memorising 8 rows needs no signal. Chapter 12 showed the same pair of numbers for a labels-shuffled run, and this is why a passing overfit test does not clear the data.
5. **Monitoring.** The log's rate column reads $2.4\times10^{-3}$ in epoch 1 and has reached $3.0\times10^{-5}$ by epoch 5, and the loss in stage 1 has stopped falling although the labels are now right. A cosine schedule that has reached its floor after 40 steps of 640 is a schedule that was told the run is 40 steps long: its horizon was given in epochs. This is the second defect: **the schedule's total is in the wrong unit**. Fix it (stage 2, where the model trains, to a validation accuracy of $0.8984$).
6. **Determinism of evaluation.** In stage 2 the validation loss is noisier than the training loss and the best epoch moves between runs of identical settings. Evaluate the same weights twice: in the shipped driver, in train mode, the two losses differ (the first of the last two rows of the checklist table above), and in eval mode they are identical to the last bit. A deterministic model evaluated twice must give the same loss. This is the third defect: **the validation pass runs in train mode**, with dropout active, drawing from the training generator (which is why the draw order is part of the stage table's description). Fix it (stage 3). Stage 3 is the healthy run of milestone 6 without early stopping, and its table agrees with the milestone-6 log at the epochs they share.

Each of these defects costs a plausible-looking run. The first is invisible to every check that takes the data as given, the second to a loss curve read in epochs, the third to anyone who reads the validation loss as exact. They are also the three most common ways a correct library produces a wrong result.

To pass, submit three things for each defect: the measurement that exposed it (the number and the checklist item), the one-line fix, and the number that shows the fix worked. Your stage-3 run must agree with the last four rows of the stage table to $10^{-4}$, and the fingerprint of the acceptance tests includes the two checks that separate stages 2 and 3.

The diagnosis itself can go wrong in four ways.

- Fixing the item that is easiest to see first. The rate column makes defect 2 obvious, but while the labels are wrong the repaired run still scores at the prior, and you conclude the schedule was not the problem.
- Reading a passing gradient check or a passing overfit test as a clean bill of health for the data.
- Comparing a stage's validation loss with the next stage's without checking that the validation pass is the same function in both.
- "Fixing" the defect by changing a setting that happens to compensate (a larger rate for the collapsed schedule). The checklist's last step is to explain the number that exposed the defect, and a compensated run does not.

## Acceptance tests

The tests are the blocks printed in the milestones above, collected here as a checklist. An implementation passes when every statement holds. "Exact" means integers or text equal; "to $t$" means the stated absolute tolerance.

| milestone | test | expected | tolerance |
|---|---|---|---|
| 1 | layer fixtures: probe loss $L$, the four entries of $y$ and $dx$, the first and last parameter gradients and their sum of squares | as printed (8 decimals) | $5\times10^{-9}$ |
| 1 | softmax cross-entropy and mean squared error on the fixture; the overflow row | as printed; every value finite | $5\times10^{-9}$ |
| 1 | dropout mask on the fixture seed | the printed pattern of 15 values | exact |
| 1 | accumulation | twice, zero-then-once and in-pieces differences below the printed bounds | $10^{-15}$ |
| 2 | every layer, input and every parameter tensor: failing entries and the directional verdict | zero failing; all pass; worst ratio below 1 | exact; ratio is a criterion, not a number to copy |
| 2 | layer norm with the centring term dropped | input fails on 15 of 15 entries; gamma and beta pass | exact |
| 2 | dimension policy | failing entries as printed for the four instances; `distinct_dims` as printed | exact |
| 2 | model checks (classifier, autoencoder) | no failing entry; directional passes | exact |
| 3 | optimizer trajectories: theta after 1, 2, 3 and 10 steps for the seven rules | as printed | $5\times10^{-9}$ |
| 3 | schedule rates; clip norm and scale | as printed (8 decimals) | $5\times10^{-9}$ |
| 4 | initialization: first three weights per scheme and shape; standard deviation | as printed | $5\times10^{-7}$ for a bit-exact generator (the standard deviation uses the divisor $n = 12288$); $3\%$ of the target for the standard deviation otherwise |
| 5 | early-stopping rule on the given losses | as printed | exact |
| 5 | resume | full checkpoint: identical; each missing piece: not identical | exact |
| 6 | task fingerprint: class counts, row 0, sum of the inputs | as printed | counts exact; $5\times10^{-7}$ |
| 6 | classifier: initial loss, the log at the shown epochs, the best and stop epochs, the checkpoint's validation and test scores | as printed | $10^{-4}$ for losses and accuracies; epochs exact |
| 6 | autoencoder: initial loss, log at the shown epochs, best epoch, test loss | as printed; equal to the table of chapter 13 | $10^{-5}$; epoch exact |
| 7 | the four stages: loss and accuracy at the shown epochs; the diagnostics table | as printed | $10^{-4}$; the sign of each reading is the test |

Two conventions apply to the whole list. A number computed from a generator started at a printed seed is reproduced only by a bit-exact generator, so test in order: the raw draws (the initialization table), then the task, then the untrained losses, then a short training run, then the long ones. A mismatch in the first column that fails points at the generator or the order of the draws; a mismatch only at the longest runs, with everything before it right, points at a summation order or a libm function (the error grows with the number of steps, but a correct implementation in 64-bit arithmetic stays within the stated tolerance for these runs). And where the table says "criterion", the statement is part of the test and not a number to copy.

A compact fingerprint of a correct library, to check at the end:

<!-- worked: project_nnlib.acceptance_fingerprint -->
| milestone | quantity | value | tolerance |
|---|---|---:|---|
| 1 | probe loss L, linear 5 -> 7 | +0.50268275 | 5e-9 |
| 1 | probe loss L, layer norm | -1.11966819 | 5e-9 |
| 1 | probe loss L, residual (5 -> 7 -> 5) | -1.98236398 | 5e-9 |
| 1 | softmax cross-entropy on the fixture | +1.67574842 | 5e-9 |
| 2 | layer-norm input entries failing without the centring term | 15 of 15 | exact |
| 3 | AdamW theta[0] after 10 steps | +0.68398993 | 5e-9 |
| 3 | schedule rate at t = 55 (hold 0) | 0.00505000 | 5e-9 |
| 4 | sd (divisor n = 12288) of He normal, 128 x 96 (stream init.he.normal) | 0.145302 | 5e-7 bit-exact; 3% of the target otherwise |
| 5 | patience 3: best epoch, stop epoch | 6, 9 | exact |
| 6 | classifier: initial training loss | 1.12218 | 1e-4 |
| 6 | classifier: best epoch, stop epoch | 19, 27 | exact |
| 6 | classifier: validation loss at the checkpoint | 0.23893 | 1e-4 |
| 6 | autoencoder: initial training loss | 1.03367 | 1e-4 |
| 6 | autoencoder: best epoch | 281 | exact |
| 6 | autoencoder: best validation loss | 0.00246 | 1e-5 |
| 7 | stage 3 equals milestone 6 without early stopping: validation loss, epoch 10 | 0.3925 | 1e-4 |
| 7 | stage 2 (eval-mode fault only): validation accuracy, epoch 40 | 0.8984 | 1e-4 |
| 7 | horizon fault only: rate of the last step, epoch 40 | 3.000e-05 | 5e-9 |
<!-- /worked -->

## The checker and optimizers in brain's own crates

The habits of this chapter are ones brain's own code enforces. `crates/gradcheck/src/lib.rs` holds the checker: `elementwise_check` and `directional_check` are the two forms of milestone 2 on real models, `dead_gradients` lists parameters whose gradient is exactly zero while the numeric one is not (the failure the tolerance gate of chapter 6 cannot see), and `zero_grad_params` finds tensors that never receive a gradient. `crates/optim/src/lib.rs` holds the optimizers of milestone 3, and `crates/gradcheck/src/clip.rs` checks the clip. `crates/toyautoencoder/src/model.rs` is the autoencoder of milestone 6 (chapter 13 describes it and its tests). What differs from this project is scale and execution, not the contract: brain's layers run as kernels on several backends and its checker runs on each of them, which Volume IV explains.

## Summary

If every acceptance test passes you have built and checked: ten layer types and two losses whose forward and backward passes agree with fixtures to $5\times10^{-9}$ (the eighth printed decimal) and with finite differences in the input and in every parameter; a checker that finds a planted fault in a layer's input gradient and a policy for the sizes it runs on; seven update rules and a schedule and clip that reproduce fixed trajectories; initializers whose draws are specified to the bit; a training loop whose checkpoints resume a run exactly, and whose every piece of state is accounted for; a classifier that reaches a test accuracy of $0.9141$ and an autoencoder that reproduces chapter 13's table from different code; and a diagnosis of three defects that each hid another.

What the project established, in order. The sizes of the test instance decide what the gradient check can see. A layer is not checked until its input gradient is. A resumed run is the same run only if the order of the examples and the state of every generator travel with the weights. A schedule is a function of steps and a loss curve is read in epochs. A passing gradient check and a passing overfit test say nothing about the data or about the evaluation.

Where this stops. The library is dense, single-threaded and written for clarity: every layer is a few loops, and the cost of the autoencoder run is the 456 multiply-adds per example of chapter 13 repeated 2400 steps over a batch of 32. It has no attention, no convolution, no batch normalization (which needs running statistics and a second mode), no gradient accumulation across calls to the optimizer and no mixed precision. The next volume trains models that are millions of times larger, where nothing can be checked by hand; the habits built here (the fixture, the checker with distinct sizes, the exact resume, the checklist in order of cost) are what stand in for checking.

## Extensions

These are optional. Each is a bigger test of the same ideas.

1. **Batch normalization.** Add the layer of chapter 10 with running statistics and the two modes. What must the checkpoint of milestone 5 now hold, and which acceptance test of milestone 5 fails if it does not?
2. **Weight decay on the right tensors.** Exclude biases, layer-norm gains and shifts from the decay. Does the classifier's test accuracy change, by how much compared with the spread over five initialization seeds, and what does your result say about the claim of chapter 11?
3. **Optimizer comparison.** Train the classifier with SGD with momentum, Adam, AdamW and Lion, each at its best rate from a sweep of seven rates on the validation set. Report the memory of each in numbers of stored floats (chapter 7) alongside the accuracies.
4. **A deeper residual stack.** Make the residual block appear 1, 4, 8 and 16 times. Compare the gradient norm per layer at initialization with and without scaling the last weights of each branch by $1/\sqrt{2L}$ (chapter 9, "Residual branches").
5. **A gradient accumulation loop.** Run the batch of 32 as four micro-batches of 8, accumulating into the same gradient before one optimizer step. Show that the log agrees with milestone 6 to $10^{-10}$ and say which layer would break this (chapter 10).
6. **Kinks.** Replace GELU by ReLU in the classifier. Which entries of the model check now fail, how many, and what does the margin test of chapter 6 say about the input that was used?
7. **A sixth planted defect.** Plant a defect of your own in the driver of milestone 7 (for example, the clip applied after the update, or the decay on the gradient), and write the table of symptoms that a reader following the checklist would see. Give the driver to another reader and ask them to find it.
8. **From fixtures to random inputs.** Replace the fixtures of the layer tests by inputs drawn from your generator and run the test 1000 times with different seeds. How many runs fail the ReLU check, and how many of those would the margin rule have rescued?

## Answers

**Recall questions.**

1. $dW_{ji} = \sum_n dy_{nj}x_{ni}$, $db_j = \sum_n dy_{nj}$ and $dx_{ni} = \sum_j dy_{nj}W_{ji}$ (chapter 5). The loss is a mean over the batch, so the layer carries no factor $1/B$. If the backward pass assigns to $dW$, a layer used twice in one model (or a batch cut into pieces, or a residual wrapper that shares a layer) keeps only the last contribution, and the gradient is wrong with no error raised.
2. With all sizes equal a transposed weight or product is shape-legal and numerically hard to tell from the correct one for some fixtures, and a mean over $B$ that should be over $B\cdot O$ is invisible when $O = 1$. The sizes should be distinct and above 1 (small primes: $B = 3$, $D = 5$, $H = 7$, $O = 2$), with at least two of everything the model sums over (chapter 6).
3. Decoupled decay first shrinks the weight to $1 - \eta\lambda\cdot 1 = 0.99$, and the Adam move of step 1 is $\eta\,\mathrm{sign}(g)$ up to $\epsilon_{\mathrm{den}}$, so $0.1$: the weight ends at $0.89$, a change of $0.11$. The coupled version adds $\lambda\theta = 0.1$ to the gradient before the moments, giving a gradient of $0.102$, and its first step is the same $0.1$, so the weight ends at $0.9$. The two differ from step 2 on, when the decay term has been divided by a gradient scale (chapter 7).
4. Step 0: $0.01\cdot 1/10 = 0.001$. Step 9: $0.01\cdot 10/10 = 0.01$, the peak. The norm of $(3, 4, 0, 12)$ is $13$, the scale is $5/(13 + 10^{-6}) = 0.3846$, and the tensors become $(1.1538, 1.5385)$ and $(0, 4.6154)$ (chapter 8).
5. A ReLU-like unit zeroes half of its input, so the sum of $n_{\mathrm{in}}$ terms must have twice the variance a linear layer would need to keep the second moment of the signal constant: $2/n_{\mathrm{in}}$, here $2/48$. The read-out feeds the loss directly and there is no later layer to rescale it, so a full-size read-out starts the loss above $\ln C$ (chapter 9, "The output layer").
6. $32\cdot 32 = 1024$ uniform draws in row-major order, a unit being kept when its draw is at least $p_{\mathrm{drop}}$ and scaled by $1/(1 - p_{\mathrm{drop}})$ (chapter 11; the generator is Volume I, chapter 17). Two evaluations are equal in eval mode, in which dropout is the identity and draws nothing; in train mode they differ because each takes the next draws.
