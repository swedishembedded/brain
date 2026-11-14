<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Project: Reproduce, Verify, Optimize, Defend

*Can you reproduce a published model component, prove it correct, make it fast and defend the result?*

Store the three weight matrices of a transformer's feed-forward block as 8-bit codes instead of 32-bit numbers, call the new block ten times in a row, and count the bytes it reads from memory: 664,576. The 32-bit block, called the same ten times, reads 23,603,200. The new block reads 35.5 times fewer bytes, every count is exact, and nothing about the arithmetic is wrong. Then decode one token through the whole model that the block belongs to, eight layers with attention and a key-value cache in front of each block and an output head at the end, and count again: 39,880,704 bytes against 26,314,752, a factor of 1.52. At a context of 2,048 cached positions the factor is 1.26. The first number is a measurement of a cache that held the 8-bit block's 0.66 MB of weights between calls and could not hold the 32-bit block's 2.36 MB; no request that decodes a token through eight layers ever sees it.

That number is one of four kinds of confident wrong answer that the work of this project produces. Reproducing a published component means turning a paper and a file of weights into a function, and a port with the up projection where the gate should be computes a function too. Proving it correct means comparing it with a reference, and a comparison by cosine cannot see a norm whose output is too small by a factor of $\sqrt{6}$. Making it cheaper means changing the numbers, and an error measured after the residual addition looks two to three times smaller than the error of the change itself. Defending the result means a claim that a later change cannot quietly undo, and a test that skips when its golden file is missing reports a pass for every regression. You build, in your own language, a system that does all four to one component, a pre-norm gated feed-forward block, and every step ends in numbers from the worked module that your build must reproduce, with planted mistakes that those numbers catch and a weaker check would not.

The steps use the whole volume: the porting workflow and the golden of Chapter 1, the traps of numerical tests of Chapter 2 (a degenerate dimension, a scale-blind metric, a gate that never runs), the valid benchmark of Chapter 3, the hypothesis stated before its test of Chapter 4, and the regression gate of Chapter 5. Five questions bring back what they rest on.

1. (Chapter 1, the porting workflow, and Volume VI, Chapter 2, reference goldens.) In what order does a disciplined port proceed from a published model to a trusted implementation, and why does its golden store the activations between the layers and not only the output?
2. (Volume VI, Chapter 2, parity tolerance.) A 32-bit computation has a longest chain of $N_{\mathrm{rnd}}$ dependent roundings. What is the tolerance of its comparison with a 64-bit reference, and what kind of mistake can never fail a comparison at that tolerance?
3. (Volume III, Chapter 6, the gradient check.) Why is the central difference preferred to the forward difference, and why is the deviation divided by $\max(1, |\text{analytic}|)$ rather than by the analytic value?
4. (Chapter 3 and Volume IV, Chapter 11, Amdahl's law.) A part that takes the share $p$ of a whole is made $g$ times cheaper. What does the whole gain, and which measurement, of the part or of the whole, says what a user pays?
5. (Chapter 4, falsifiable hypotheses, and Chapter 5, regression gates.) What must be written down before a measurement for its result to be able to refute a claim, and what makes a gate that skips its check worse than no gate?

## Seeds, roundings and digests

Every random input comes from the book's SplitMix64 generator ([Volume I, Chapter 17](../v01-mathematics/17-randomness.md)) started at the seed `seed_of(name)` printed beside its table, with `uniform()` and `integer(n)` as the pseudocode page defines them; each named generator is started once and never restarted, and the order of its draws is printed with it. Every weight and every input is rounded to the nearest 32-bit number (ties to even) as it is drawn, so that a 32-bit implementation and a 64-bit one start from the same bytes. Two arithmetics are used. The 64-bit one is ordinary double precision. The 32-bit one, which the port uses, rounds the result of every addition, subtraction, multiplication, division and square root to 32 bits, one operation at a time as the pseudocode page defines, with no fused multiply-add; `exp` is the library's 64-bit value rounded once to 32 bits, and the constant $\epsilon = 10^{-6}$ is rounded to 32 bits before it is used. A sum starts from its first term and adds the others left to right, in increasing index; a dot product rounds its first product, then each further product and each addition. Matrices are row-major with shape [outputs, inputs], indices count from 0, and positions are processed one at a time and independently.

The tables print errors to three significant digits. Nothing in this project is trained or sampled chaotically, so every printed digit is stable; the only library function behind a printed number is `exp`, whose last bit can differ between libraries, and no printed digit depends on it. Digests are the 64-bit FNV-1a hash of Volume XIII, Chapter 7, over the bytes of the 32-bit numbers in little-endian order and in the order stated. The worked module is `scripts/book/worked/project_reproduction.py`; it imports the 32-bit rounding, the three comparison measures and the ordered sums of Volume VI, Chapter 2 (`importing_models`), the digest of Volume XIII, Chapter 7 (`checkpoints_and_recovery.fnv1a64`) and the 95 percent interval of Volume II, Chapter 8 (`experiments._interval`), and implements the rest itself.

## The block as published

The component is the feed-forward half of a modern decoder layer, as its two papers describe it. Zhang and Sennrich (2019) normalize a vector by its root mean square without subtracting a mean; Shazeer (2020) replaces the single hidden projection of a feed-forward block by two, one of which passes through a smooth gate. For an input row $\mathbf{x}$ of width $D$, a gain $\mathbf{g}$ of width $D$, two matrices $\mathbf{W}_{\mathrm{gate}}, \mathbf{W}_{\mathrm{up}}$ of shape $[F, D]$ and one $\mathbf{W}_{\mathrm{down}}$ of shape $[D, F]$:

$$
\mathbf{n} = \mathbf{g} \odot \frac{\mathbf{x}}{\sqrt{\tfrac{1}{D}\sum_i x_i^2 + \epsilon}}, \qquad
\mathbf{h} = \mathrm{silu}(\mathbf{W}_{\mathrm{gate}}\mathbf{n}) \odot (\mathbf{W}_{\mathrm{up}}\mathbf{n}), \qquad
\mathbf{y} = \mathbf{x} + \mathbf{W}_{\mathrm{down}}\mathbf{h},
$$

with $\mathrm{silu}(z) = z\,\sigma(z)$ and $\epsilon = 10^{-6}$. Write $\mathbf{f} = \mathbf{W}_{\mathrm{down}}\mathbf{h}$ for the branch that the residual addition joins to $\mathbf{x}$. Every quantity has a name, and the six that the golden will store are the taps: $\mathbf{n}$, the gate projection, the up projection, $\mathbf{h}$, $\mathbf{f}$ and $\mathbf{y}$.

![A left-to-right diagram of the block. The input x of width 6 enters an RMS norm with gain g, giving the tap n. n feeds W_gate (10 x 6) and W_up (10 x 6), giving the taps gate and up of width 10; the SiLU of gate times up gives the tap h of width 10; W_down (6 x 10) gives the tap f of width 6; adding x along a dashed residual path gives the tap y. The six taps are shaded ovals.](img/reproduction-component.svg)

The printable component has $D = 6$, $F = 10$ and $T = 5$ positions: three sizes that differ from each other and from 1, so that a mistake that confuses two of them changes a shape or a number (the test-dimension choice of Volume III, Chapter 6). The weights are drawn at the scales of a freshly initialized block: the gain near 1, the matrices uniform with a bound of one over the square root of their input width.

<!-- worked: project_reproduction.component -->
The weights come from the generator `project_reproduction.weights` (seed 13960080960163087327), the inputs from `project_reproduction.inputs` (seed 9232453125161690418). Each number is one uniform() u put through the rule of its row and then rounded to the nearest 32-bit number (ties to even); the square roots are the correctly rounded 64-bit ones. The weights are drawn in the order of the rows, each matrix row-major ([outputs, inputs]), so the gain takes draws 0 to 5, W_gate 6 to 65, W_up 66 to 125 and W_down 126 to 185; the inputs are drawn position by position. The digest is the FNV-1a 64 of the 32-bit numbers, little-endian, in that order. The second set of inputs, the probe, is each input times 0.001 rounded to 32 bits.

| tensor | shape | rule | numbers | FNV-1a 64 |
|---|---|---|---:|---|
| gain g | [6] | 1 + 0.25 (2u - 1) | 6 | d47b8477f725facc |
| W_gate | [10, 6] | (2u - 1) / sqrt(6) | 60 | 79d32c86d5f7bae2 |
| W_up | [10, 6] | (2u - 1) / sqrt(6) | 60 | 6268c9a1ad864941 |
| W_down | [6, 10] | (2u - 1) / sqrt(10) | 60 | 16fbf043434d3133 |
| inputs x | [5, 6] | 2u - 1 | 30 | cc9e25d6b7827b75 |

The first three gains are 0.84413940, 1.07378829, 1.02245903; the first three entries of row 0 of W_gate are 0.16970731, -0.33408710, -0.25887522; the first three inputs of position 0 are -0.64859658, -0.19515263, -0.49365738.
<!-- /worked -->

The publisher does not ship the matrices in the port's layout, and few publishers do. The file holds three tensors: `ffn_norm.weight` of shape [6], the gain; `ffn.gate_up.weight` of shape [20, 6], the gate matrix and the up matrix stacked into one, gate rows first; and `ffn.down.weight` of shape [10, 6], the down projection stored input-major, so that its entry $[j, i]$ is $W_{\mathrm{down}}[i, j]$. The import is a weight-name mapping (Volume VI, Chapter 2): rename the gain, cut the fused matrix at row $F$, transpose the down projection, and check every shape before anything runs.

Your build has its component when it draws the 186 weights and 30 inputs in the stated order and reproduces the five digests and the printed values above.

## A reference in 64 bits, and the golden it leaves

The reference is the straightforward implementation of the three equations in 64-bit arithmetic, written from the papers and nothing else, and the golden is what it leaves behind: the inputs and all six taps of every position, each rounded once to 32 bits and stored under a name. The golden generation is the step where the reference's output stops being a run and becomes evidence, so it is done by a script that records which weights produced it (here, the digests of the previous table) and is run again whenever they change, never edited by hand.

One set of inputs is not enough, for a reason that the norm makes plain. The constant $\epsilon$ is added to a mean square near 0.33 for inputs drawn from $[-1, 1)$, where it changes the result by about 1.5 parts in a million; a mistake about where $\epsilon$ goes can then be smaller than the rounding of the port and invisible to every comparison. The golden therefore holds a second set, the probe: the same inputs times $10^{-3}$, whose mean squares are near $3 \times 10^{-7}$, below $\epsilon$. A golden designed this way puts its inputs where the mistakes it must catch are large, the way Volume VI, Chapter 2 shrank its activations until a wrong epsilon showed.

<!-- worked: project_reproduction.golden_contents -->
The golden: the 64-bit reference run on both sets of inputs, every tap rounded once to 32 bits. Row-major, positions first; the digest is the FNV-1a 64 of the 32-bit numbers, little-endian.

| tensor | shape | what | FNV-1a 64 |
|---|---|---|---|
| x | 5 x 6 | the unit inputs | cc9e25d6b7827b75 |
| x_probe | 5 x 6 | the probe inputs (x times 0.001) | 60864dfc251dd9e5 |
| unit/n | 5 x 6 | output of the norm | cfe1e67d1316e80e |
| unit/gate | 5 x 10 | gate projection | 4e5641d02047fcf6 |
| unit/up | 5 x 10 | up projection | d56795ce3b99c5ed |
| unit/h | 5 x 10 | SiLU(gate) times up | e136e3f130a67d41 |
| unit/f | 5 x 6 | down projection | c4d4d53c6e47117b |
| unit/y | 5 x 6 | block output, residual added | a0c4a764b500fd41 |
| probe/n | 5 x 6 | output of the norm | 971a0e38558caf50 |
| probe/gate | 5 x 10 | gate projection | 12ebeae432e2eaa1 |
| probe/up | 5 x 10 | up projection | 9da9eea7f91b285c |
| probe/h | 5 x 10 | SiLU(gate) times up | 4e9edfd720900394 |
| probe/f | 5 x 6 | down projection | 0a8d60b10ca96b4f |
| probe/y | 5 x 6 | block output, residual added | 440f1dc3f05d816b |

Row 0 of `unit/y` is -0.644807, -0.195842, -0.622347, 1.044123, -0.640250, -0.604510.
<!-- /worked -->

The golden is complete when your reference, run on your inputs, reproduces the fourteen digests. A digest that differs in the inputs points at the generator or its rounding; one that differs only in the taps points at the reference, and the tap where it first differs says which equation.

## Reading the publisher's tensors

A mapping can be wrong in two ways that a loader feels very differently. A missed transpose of the down projection gives a matrix of shape [10, 6] where the port expects [6, 10], and the shape check refuses it by name before a single number is computed. A cut of the fused matrix in the wrong order gives two matrices of shape [10, 6] each, exactly the shapes the port expects, and the block loads and runs and computes another function.

<!-- worked: project_reproduction.import_checks -->
The mapping of the published tensors onto the port, at the 6 x 10 component and at a square 8 x 8 one (weights from `project_reproduction.square` (seed 1622393052264195514), inputs from `project_reproduction.square_inputs` (seed 14767466789678329543), 5 positions, the same rules and draw order). rel_l2 compares the port's y with the unrounded 64-bit reference of the same weights (not with the golden, whose y is rounded to 32 bits).

| configuration and mapping | shape check | rel_l2 of y |
|---|---|---:|
| 6 x 10, the mapping as specified | loads | 5.97e-08 |
| 6 x 10, gate and up cut in the wrong order | loads | 7.65e-02 |
| 6 x 10, the down projection not transposed | refused: Wd has shape [10, 6], the port expects [6, 10] | none |
| 8 x 8, the mapping as specified | loads | 3.10e-08 |
| 8 x 8, the down projection not transposed | loads | 2.06e-01 |
<!-- /worked -->

The fourth and fifth rows are the degenerate-dimension trap of Chapter 2 in its plainest form. On a square component with $D = F = 8$, the matrix that should be transposed has the same shape either way, the shape check passes, and the only witness is the output, which is 21 percent off. Had the component of this project been square, the shape check would have passed a missed transpose, and every later step would have rested on the comparison alone. The 6 x 10 component makes the loader itself refuse that mistake, and leaves only the swapped cut for the comparison to find.

Your import is right when the specified mapping loads and the port's output is within $6 \times 10^{-8}$ of the reference (the first row), and wrong in the way the table says when you plant each mistake: refused by name for the missed transpose at 6 x 10, loaded with a relative error of 0.0765 for the swapped cut.

## A bound for every tap

The port is the same three equations in 32-bit arithmetic, one operation at a time. Its pseudocode fixes every order, because a different order is a different 32-bit result:

```text
function dot_lr(a: array<real>[n], b: array<real>[n]) -> real:
    require n >= 1, "an empty product"
    s = a[0] * b[0]
    for i in 1 .. n:
        s = s + a[i] * b[i]
    return s

function block_forward(x: array<real>[D], g: array<real>[D], Wg: array<real>[F, D],
                       Wu: array<real>[F, D], Wd: array<real>[D, F], eps: real) -> Taps:
    inv = 1 / sqrt(dot_lr(x, x) / D + eps)
    n = zeros(real, [D])
    for i in 0 .. D:
        n[i] = g[i] * (x[i] * inv)
    gate = zeros(real, [F])
    up = zeros(real, [F])
    h = zeros(real, [F])
    for j in 0 .. F:
        gate[j] = dot_lr(Wg[j, :], n)
        up[j] = dot_lr(Wu[j, :], n)
        h[j] = (gate[j] * sigmoid(gate[j])) * up[j]
    f = zeros(real, [D])
    y = zeros(real, [D])
    for i in 0 .. D:
        f[i] = dot_lr(Wd[i, :], h)
        y[i] = x[i] + f[i]
    return Taps(n, gate, up, h, f, y)
```

`Taps` is the record of the six arrays, one call handles one position, and `sigmoid` is the routine of the pseudocode page, `1 / (1 + exp(-z))` for $z \ge 0$ and `exp(z) / (1 + exp(z))` otherwise, three roundings either way; `inv` is formed as the pseudocode reads, the division by $D$, the addition of `eps`, the square root and the reciprocal each rounded. Run in 64 bits, `block_forward` is the reference; run in 32 bits, it is the port.

The tolerance of each tap comes from the count of roundings, the parity tolerance of Volume VI, Chapter 2: on the longest chain of dependent operations to a tap, a sum of $n$ products costs $n$ roundings and every further operation one, and the relative error of a tap is bounded, to first order and when no stage amplifies the error of its input, by that count times $u = 2^{-24}$. The norm costs $D + 6$ (the sum of $D$ squares, the division, the addition of $\epsilon$, the square root, the reciprocal, the product with the reciprocal and the product with the gain). A projection adds $D$; the SiLU and the product add 5 (exp, the addition, the division, the product with the gate, the product with up); the down projection adds $F$; the residual adds 1. The golden's own rounding to 32 bits adds one more to every tap, so the bound of a tap with $N_{\mathrm{rnd}}$ roundings is $(N_{\mathrm{rnd}} + 1)\,2^{-24}$.

<!-- worked: project_reproduction.tap_parity -->
The 32-bit port against the golden, tap by tap. 'Roundings' counts the longest chain of dependent operations to the tap (a sum of n products costs n, each further operation 1); the bound is that count plus one (the golden's own rounding to 32 bits) times 2^-24. The port rounds every product, addition, division, square root and exp to 32 bits, and rounds eps = 1e-6 to 32 bits before use; its sums run left to right.

| tap | roundings | bound | rel_l2, unit inputs | rel_l2, probe inputs |
|---|---:|---:|---:|---:|
| n | 12 | 7.75e-07 | 7.36e-08 | 8.11e-08 |
| gate | 18 | 1.13e-06 | 1.14e-07 | 8.71e-08 |
| up | 18 | 1.13e-06 | 8.58e-08 | 9.76e-08 |
| h | 23 | 1.43e-06 | 2.65e-07 | 1.29e-07 |
| f | 33 | 2.03e-06 | 2.77e-07 | 1.71e-07 |
| y | 34 | 2.09e-06 | 6.79e-08 | 1.66e-07 |
<!-- /worked -->

The measured errors sit between 5 and 31 times below their bounds on both sets of inputs. This is the layer-by-layer parity of Chapter 1 in its chained form, the port running on its own outputs under a bound that grows along the chain: not one comparison of the output, but a comparison at every tap against a bound derived from the arithmetic, so that a failure names the first equation that is wrong. (The isolated form would feed each stage the golden's own input; for a block of three equations the chained form already names every planted mistake below.) The bound is a theorem under its assumption that no stage amplifies error, and the measurement is the evidence that the assumption held here.

Your port passes when every tap is within its bound on both sets of inputs. The digits of your errors may differ from the table's in the last place if your `exp` differs from the library this table used in its last bit, but not by a factor that brings any of them near a bound.

## Four mistakes, and the tap that names each

The comparison earns its cost on mistakes. Each row below is the port with one mistake planted, run on both sets of inputs, and the search of Volume VI, Chapter 2 for the first tap above its bound:

<!-- worked: project_reproduction.mistakes -->
Four planted mistakes, each run through the 32-bit port on both sets of inputs. 'First tap above' is the first tap, in the order n, gate, up, h, f, y, whose rel_l2 against the golden exceeds its bound; the next two columns measure that tap, and the last one the block output. 1 - cosine is printed as 'below 1e-9' when it is.

| mistake | inputs | first tap above | rel_l2 there | 1 - cosine there | rel_l2 of y |
|---|---|---|---:|---:|---:|
| gate and up cut in the wrong order | unit | gate | 1.51e+00 | 1.30e+00 | 7.65e-02 |
| gate and up cut in the wrong order | probe | gate | 1.52e+00 | 1.30e+00 | 2.57e-01 |
| mean of the squares replaced by their sum | unit | n | 5.92e-01 | below 1e-9 | 1.51e-01 |
| mean of the squares replaced by their sum | probe | n | 3.81e-01 | 4.65e-04 | 6.38e-01 |
| eps added after the square root | unit | none |  |  | 1.69e-07 |
| eps added after the square root | probe | n | 7.70e-01 | 1.21e-03 | 2.12e+00 |
| sigmoid of the gate without the product by the gate (GLU for SwiGLU) | unit | h | 1.17e+00 | 3.57e-01 | 1.74e-01 |
| sigmoid of the gate without the product by the gate (GLU for SwiGLU) | probe | h | 2.43e+00 | 4.99e-01 | 1.62e+00 |
<!-- /worked -->

Three of the four mistakes are found at the tap where they are made. The swapped cut shows at the gate projection, with a relative error of 1.51 on the unit inputs; the norm that sums where it should average shows at its own output; the GLU in place of the SwiGLU shows at $\mathbf{h}$. The sum in the norm is also the scale-invariant-metric trap of Chapter 2: on the unit inputs its output is the right output times very nearly $1/\sqrt{6}$, a relative error of 0.592 whose cosine with the golden is 1 to nine places, so a comparison by cosine at a floor of 0.9999 passes it at the very tap that is wrong. The error is visible downstream, 0.151 at $\mathbf{y}$, but a cosine ladder would have blamed whichever later stage first lost its direction.

The fourth mistake, $\epsilon$ added after the square root, passes every bound on the unit inputs. Its effect there is a change of the norm's divisor from $\sqrt{m + \epsilon}$ to $\sqrt{m} + \epsilon$, which for a mean square $m$ near 0.33 differs by $\epsilon(1 - 1/(2\sqrt{m}))$, about $1.3 \times 10^{-7}$ in a divisor near 0.58, a relative change of about $2.3 \times 10^{-7}$: less than the $7.75 \times 10^{-7}$ that the bound of the norm allows for rounding. On the probe inputs the two divisors differ by a factor near 2, and the mistake shows at the norm with an error of 0.770. No tolerance could have caught it on the unit inputs without also failing the correct port; the second set of inputs is what catches it, and that set exists because the golden was designed around the constant.

Your mistakes are planted correctly when each reproduces its row: the first tap above, the error there and the error of $\mathbf{y}$, to the digits printed.

## The backward pass against central differences

A port that will only ever run forward is finished at this point. A component that will be fine-tuned also needs its backward pass, and the backward pass has no golden: the publisher's file holds weights, not gradients. The check is the gradient check of Volume III, Chapter 6, in 64-bit arithmetic, on the scalar loss $\sum_t \sum_i c_{ti}\, y_{ti}$ with fixed random weights $c_{ti}$, which sends a different gradient into every output.

The rules come from the chain rule taken one equation at a time, last to first, all in 64 bits; the sums in the pseudocode below run in increasing index, and the order does not matter at the criterion of the check. The gradient arriving at $\mathbf{y}$ is $\mathbf{c}$, and since $\mathbf{y} = \mathbf{x} + \mathbf{f}$ it passes unchanged to both $\mathbf{f}$ and $\mathbf{x}$. The down projection gives its weight gradient as an outer product and passes $\mathbf{W}_{\mathrm{down}}^\top$ times the gradient to $\mathbf{h}$. The product $h_j = \mathrm{silu}(a_j)\, b_j$ splits it into $b_j\, \mathrm{silu}'(a_j)$ for the gate and $\mathrm{silu}(a_j)$ for up, where the slope of the SiLU is $\sigma(a)(1 + a(1 - \sigma(a)))$, the `silu_slope` of the pseudocode page. The two projections give outer products and send the sum of their transposed products to $\mathbf{n}$, and the norm's own backward rule is `rms_norm_backward` of the pseudocode page, whose gradient for $\mathbf{x}$ carries a term through the mean square because every input moves the divisor:

```text
function block_backward(c: array<real>[T, D], x: array<real>[T, D], g: array<real>[D],
                        Wg: array<real>[F, D], Wu: array<real>[F, D],
                        Wd: array<real>[D, F]) -> Grads:
    dx = zeros(real, [T, D])
    dg = zeros(real, [D])
    dWg = zeros(real, [F, D])
    dWu = zeros(real, [F, D])
    dWd = zeros(real, [D, F])
    for t in 0 .. T:
        (y1, xhat, inv) = rms_norm(x[t : t + 1, :], g, 1e-6)     # one row: shapes [1, D], [1, D], [1]
        n = y1[0, :]
        a = zeros(real, [F])
        b = zeros(real, [F])
        h = zeros(real, [F])
        for j in 0 .. F:
            a[j] = dot(Wg[j, :], n)
            b[j] = dot(Wu[j, :], n)
            h[j] = (a[j] * sigmoid(a[j])) * b[j]
        dh = zeros(real, [F])
        for i in 0 .. D:
            for j in 0 .. F:
                dWd[i, j] = dWd[i, j] + c[t, i] * h[j]
                dh[j] = dh[j] + c[t, i] * Wd[i, j]
        dn = zeros(real, [1, D])
        for j in 0 .. F:
            da = dh[j] * b[j] * silu_slope(a[j])
            db = dh[j] * (a[j] * sigmoid(a[j]))
            for i in 0 .. D:
                dWg[j, i] = dWg[j, i] + da * n[i]
                dWu[j, i] = dWu[j, i] + db * n[i]
                dn[0, i] = dn[0, i] + da * Wg[j, i] + db * Wu[j, i]
        (dxn, dgt) = rms_norm_backward(dn, xhat, inv, g)
        for i in 0 .. D:
            dx[t, i] = c[t, i] + dxn[0, i]
            dg[i] = dg[i] + dgt[i]
    return Grads(dx, dg, dWg, dWu, dWd)
```

<!-- worked: project_reproduction.gradient_check -->
The backward pass against central differences, in 64-bit arithmetic, on the component of the tables above and its unit inputs. The loss is the sum over positions t and coordinates i of c[t][i] y[t][i], with c drawn from `project_reproduction.probe` (seed 16796600673050480404) as 2u - 1, position by position (no rounding to 32 bits). Each of the 216 numbers (30 inputs, 6 gains, 60 + 60 + 60 weights) is moved by +1e-6 and -1e-6; the deviation is |numeric - analytic| / max(1, |analytic|) and the criterion is that its largest value is below 1e-6. 'Where' is the number with the largest deviation, [row, column] counted from 0.

| backward pass | largest deviation | where |
|---|---:|---|
| backward as specified | passes (below 1e-6) |  |
| SiLU slope taken as sigmoid(gate) | 1.94e-01 | W_gate[4, 1] |
| norm backward without the mean term | 2.71e-01 | x[4, 1] |
<!-- /worked -->

Two classic backward mistakes are planted. Taking the slope of the SiLU as $\sigma(a)$, the slope of the gate alone, forgets that the gate value $a$ also multiplies the sigmoid, and the check finds it with a deviation of 0.194 in a weight of $\mathbf{W}_{\mathrm{gate}}$. Dropping the term through the mean square treats the divisor as a constant, and the check finds it with 0.271 in an input. Neither mistake changes the forward pass, so no golden could have caught either; the check is the only witness the backward pass has.

Your backward pass is right when it passes the check on all 216 numbers, and your check is right when each planted mistake fails it with the deviation and at the number of its row.

## Eight bits per weight, a scale per thirty-two

The component is now reproduced and verified, and the cheaper version can be judged against it. Decoding a token reads every weight of the block once and does two FLOPs with it, so the block reads far more bytes than its arithmetic can use (Volume IV, Chapter 3). Storing each weight in one byte instead of four removes three quarters of those bytes, at the price of a rounding of every weight.

The scheme stores, for every group of $G = 32$ consecutive weights of a row, one 32-bit scale $s = \max(|w|, 10^{-8})/127$ over the group, and for every weight the integer code $q = \mathrm{round}(w/s)$, rounded half away from zero and clamped to $[-127, 127]$; a weight is read back as $s\,q$. A product with a vector multiplies the codes of each group by the vector's entries, sums them, multiplies that partial sum by the group's scale and adds the groups. Both are short enough to write out:

```text
function quantize_rows(W: array<real>[m, k], G: int) -> (array<real>[m, k], array<real>[m, k // G]):
    require k mod G == 0, "a row must hold whole groups"
    Q = zeros(real, [m, k])
    S = zeros(real, [m, k // G])
    for o in 0 .. m:
        for gi in 0 .. k // G:
            a = 0.0
            for j in gi * G .. gi * G + G:
                a = max(a, abs(W[o, j]))
            S[o, gi] = max(a, 1e-8) / 127
            for j in gi * G .. gi * G + G:
                Q[o, j] = min(127, max(-127, round(W[o, j] / S[o, gi], 0)))
    return (Q, S)

function int8_matvec(Q: array<real>[m, k], S: array<real>[m, k // G], x: array<real>[k],
                     G: int) -> array<real>[m]:
    out = zeros(real, [m])
    for o in 0 .. m:
        acc = 0.0
        for gi in 0 .. k // G:
            part = 0.0
            for j in gi * G .. gi * G + G:
                part = part + Q[o, j] * x[j]
            acc = acc + S[o, gi] * part
        out[o] = acc
    return out
```

`round(v, 0)` is the pseudocode page's rounding to an integer with ties away from zero; the codes are whole numbers from -127 to 127, held here as `real` and stored as one signed byte each. Every operation is in 64 bits in this project, so that the measured error is the error of the codes and not of a narrow arithmetic. One scale per row is the same routine with $G = k$.

The work of a block for one token is counted by a stated rule, and the claims below rest on it. The component of the claims is larger than the printable one, $D = 256$ and $F = 768$ (both multiples of 32, so every row holds whole groups), because a block of six numbers has no memory traffic worth counting.

<!-- worked: project_reproduction.work_counts -->
One feed-forward block at D = 256, F = 768, one token. FLOPs: a sum of n products is n multiplications and n - 1 additions; the norm is 4D + 3 (D squares, D - 1 additions, a division by D, the addition of eps, a square root, a reciprocal, D products by the reciprocal and D by the gain); the SiLU and the product with up are 5 per hidden element (exp, add, divide, two products; the negation is exact and free); the residual is D additions. An 8-bit product multiplies each group's partial sum by its scale and adds the groups, which adds one FLOP per group of each row; turning a code into a number is exact and not counted. Bytes: every weight, code, scale and gain read once from memory; the vectors of D and F numbers are taken to stay in cache and count 0.

| weights | FLOPs | bytes | FLOPs per byte |
|---|---:|---:|---:|
| 32-bit weights | 1,182,979 | 2,360,320 | 0.5012 |
| 8-bit codes, a scale per 32 | 1,201,411 | 664,576 | 1.8078 |
| 8-bit codes, a scale per row | 1,184,771 | 598,016 | 1.9812 |

The ratio of the bytes of the first two rows is 3.5516; the vectors left out (x, n, gate, up, h, f, y: 3328 numbers, 13,312 bytes if each were read or written once) are 2.00 percent of the 8-bit block's bytes.
<!-- /worked -->

The 8-bit block does 18,432 more FLOPs than the 32-bit one (one scale product and one addition per group of every row) and reads 1,695,744 fewer bytes. Its byte ratio, 3.5516, is not 4: each group of 32 one-byte codes carries a 4-byte scale, 1.125 bytes per weight, and the block keeps its 32-bit gain. The per-row scheme reads fewer bytes still, and its error on weights with outliers is why it is not used.

## Ten calls that measured the cache

The opening's 35.5 comes from a benchmark of the kind Chapter 3 warns about, and the simulation that produced it makes the mechanism plain. The simulated cache is a one-level model with a strict least-recently-used rule, a simplification of any real memory system: it holds 16,384 lines of 64 bytes (1 MiB). A read of a line that the cache holds moves nothing; a read of any other line moves 64 bytes from memory and evicts the least recently used line when the cache is full. A write goes to memory and counts its own bytes. Every array has its own range of addresses, starting on a line, and is read line by line in address order.

<!-- worked: project_reproduction.cache_trap -->
Bytes read from memory, by a simulated cache of 16,384 lines of 64 bytes (1 MiB), least recently used line out first; a write goes to memory and counts its own bytes. 'Ten calls' runs one block ten times in a row on the same weights (gain, then the matrices, codes and scales in the order of the work table, each read line by line). 'Whole step' decodes the token at context 512 (positions 0 to 511 cached) through the decoder of 8 layers, after two steps at contexts 510 and 511 that warmed the cache: the embedding row of position (position mod 4096), then per layer the attention gain, W_q, W_k, W_v (256 x 256, 32-bit), the K and V caches (256 32-bit numbers per position), W_o and the feed-forward block, with K and V of the new position written (2,048 bytes), then the final gain and the output head (4096 x 256, 32-bit). Each array has its own address range starting on a line. 'Counted' is the same step by the rule of the work table.

| weights | one block, counted | ten calls, simulated | whole step, simulated | whole step, counted |
|---|---:|---:|---:|---:|
| 32-bit weights | 2,360,320 | 23,603,200 | 39,880,704 | 39,880,704 |
| 8-bit codes, a scale per 32 | 664,576 | 664,576 | 26,314,752 | 26,314,752 |

Ratios of 32-bit to 8-bit: one block's bytes 3.5516, ten calls 35.52, the whole step 1.5155.
<!-- /worked -->

The 32-bit block's 2,360,320 bytes are larger than the cache, and reading them in the same order every call evicts each line before it is needed again: every call misses on every line, ten calls read ten blocks. The 8-bit block's 664,576 bytes fit, so the first call reads them and the next nine read nothing. The ratio of the ten calls, 35.52, is the ratio of a block that does not fit in a cache to one that does, repeated ten times; it would be 3.55 at one call and grows without bound with the number of calls. A decoding step reads eight blocks, eight sets of attention weights, their caches and the output head, 26 to 40 MB in all, and every line of it is gone before the next token wants it: the simulated step and the counted step agree to the byte, and the ratio is 1.5155. The whole step is the unit of work a token pays for in the model, with everything the model reads inside the bracket; it is a step towards the end-to-end benchmark of Chapter 3, which would also drive requests through the server's entry point with their batching and scheduling, and it is already enough to refute the microbenchmark. The lesson of the microbenchmark is not that it is false, since it counts its own bytes exactly, but that it answers a question no request asks.

Your simulation is right when it reproduces the four simulated numbers of the table exactly, and your counting rule when it reproduces the counted ones.

## The whole step and the context it depends on

The whole step is not one number either. Of the bytes of a 32-bit decoding step, the feed-forward weights are a share $p$, and the 8-bit change divides only that share by the block's ratio $g = 3.5516$: by Amdahl's law (Volume IV, Chapter 11), stated here in bytes instead of time, the step's ratio is $1/((1 - p) + p/g)$. The count of the step makes $p$ explicit. With 32-bit weights,

$$
N_{\mathrm{byte}}(S) = 4D + L\,(4D + 16D^2 + 8SD + 8D + B_{\mathrm{ffn}}) + 4D + 4VD,
$$

for the embedding row, $L$ layers of an attention gain, four $D \times D$ matrices, a key and a value cache of $S$ positions read and one position of each written, and the block's $B_{\mathrm{ffn}}$ bytes, then the final gain and the $V \times D$ output head, with $D = 256$, $F = 768$, $L = 8$ and $V = 4{,}096$. $B_{\mathrm{ffn}}$ is $4D + 12FD$ for 32-bit weights and $4D + 3FD + 12FD/32$ for 8-bit codes with a scale per 32. Only the cache term grows with $S$, so the share of the feed-forward weights falls as the context grows, and the ratio with it.

<!-- worked: project_reproduction.context_sweep -->
Bytes of one decoding step of the 8-layer decoder by the counting rule, against the context S (the number of cached positions): 32-bit weights everywhere, and the feed-forward blocks in 8-bit codes with a scale per 32. 'FFN share' is the part of the 32-bit step that the feed-forward weights take; the Amdahl column is 1 / ((1 - p) + p / g) with that share p and g the byte ratio of one block (3.5516). 'At least 1.4' compares the ratio with the claim.

| context S | 32-bit, bytes | 8-bit, bytes | ratio | FFN share | Amdahl | at least 1.4 |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 31,492,096 | 17,926,144 | 1.7568 | 0.5996 | 1.7568 | yes |
| 128 | 33,589,248 | 20,023,296 | 1.6775 | 0.5622 | 1.6775 | yes |
| 512 | 39,880,704 | 26,314,752 | 1.5155 | 0.4735 | 1.5155 | yes |
| 975 | 47,466,496 | 33,900,544 | 1.4002 | 0.3978 | 1.4002 | yes |
| 976 | 47,482,880 | 33,916,928 | 1.4000 | 0.3977 | 1.4000 | no |
| 1,024 | 48,269,312 | 34,703,360 | 1.3909 | 0.3912 | 1.3909 | no |
| 2,048 | 65,046,528 | 51,480,576 | 1.2635 | 0.2903 | 1.2635 | no |
| 8,192 | 165,709,824 | 152,143,872 | 1.0892 | 0.1139 | 1.0892 | no |

The largest context at which the ratio is still at least 1.4 is S = 975.
<!-- /worked -->

The Amdahl column equals the ratio in every row, as it must when the parts of the cost add, and it shows what the change can and cannot do. With no context at all the step's ratio is 1.7568, about half of the block's; at a context of 2,048 positions it is 1.2635; at 8,192 it is 1.0892, and the key-value cache, which the change does not touch, is most of what the step reads.

![The ratio of the bytes of a 32-bit decoding step to those of a step with 8-bit feed-forward weights, against the context from 0 to 8,192 cached positions. The ratio starts at 1.76, falls through the dashed line of the claim, 1.4, at a context of 975, and approaches 1.09 at 8,192. A dashed horizontal line at 3.55 marks the ratio of one block alone, which no context reaches.](img/byte-ratio-context.svg)

## Batches, ridges and the spec sheet

A server that decodes $B$ tokens at once reads the weights once for all of them and does $B$ times the arithmetic, so the block's FLOPs per byte grow with $B$ until the device's arithmetic, not its memory, sets the cost. The roofline of Volume IV, Chapter 3 says where: a kernel whose FLOPs per byte are below the ridge point $\mathcal{I}_{\mathrm{ridge}}$ (the peak FLOP rate over the bandwidth) is bound by its bytes, and above it by its FLOPs. Measured in bytes, the cost bound of a kernel is $\max(N_{\mathrm{flop}}/\mathcal{I}_{\mathrm{ridge}}, N_{\mathrm{byte}})$, and the ratio of two bounds is the most a change can gain. It depends on the ridge, and the ridge depends on whether it was measured or read from a spec sheet, the distinction between measured and nominal peak of Chapter 3.

<!-- perf-number: historical measurement, commit 73466fee0 -->
The two ridges of the table are those of one GPU as the message of commit `73466fee0` (6 August 2026) records them, and are quoted from it, not re-measured here: 34.0 FLOPs per byte from the spec sheet, and 36.7 from the arithmetic and bandwidth roofs that brain measured on the card, which reached 89.8 and 83.1 percent of their spec-sheet values.

<!-- worked: project_reproduction.ridge_table -->
One feed-forward block applied to B tokens at once (the weights read once, the FLOPs of the work table times B). The cost bound of a kernel is max(FLOPs / ridge, bytes), in units of bytes, where the ridge is the device's peak FLOP rate divided by its bandwidth; the ratio of the two bounds is the most the 8-bit block can gain. Ridges: 34.0 FLOPs per byte from the spec sheet and 36.7 from the measured roofs of one device.

| tokens B | FLOPs per byte, 32-bit | FLOPs per byte, 8-bit | bound ratio, nominal ridge | bound ratio, measured ridge |
|---|---:|---:|---:|---:|
| 1 | 0.50 | 1.81 | 3.5516 | 3.5516 |
| 8 | 4.01 | 14.46 | 3.5516 | 3.5516 |
| 16 | 8.02 | 28.92 | 3.5516 | 3.5516 |
| 20 | 10.02 | 36.16 | 3.3399 | 3.5516 |
| 24 | 12.03 | 43.39 | 2.7832 | 3.0042 |
| 32 | 16.04 | 57.85 | 2.0874 | 2.2532 |
| 64 | 32.08 | 115.70 | 1.0437 | 1.1266 |
<!-- /worked -->

Up to 16 tokens both blocks are bound by their bytes and the ratio is the block's 3.5516 under either ridge. At 20 tokens the 8-bit block does 36.16 FLOPs per byte, above the nominal ridge and below the measured one: the spec sheet predicts that its arithmetic has become the limit and that the gain has fallen to 3.3399, while the measured roofs say it is still bound by its bytes and gains the full 3.5516. From 24 tokens on both ridges put the 8-bit block above the ridge, and the measured ridge predicts gains 8 percent larger than the nominal one. The direction is not one a reader would guess: the bandwidth fell further below its spec-sheet value than the arithmetic did, so the measured device is more bound by memory than its spec sheet says, and an optimization that saves bytes is worth more on it. A prediction made with the nominal ridge is wrong in a direction that only the measurement could have told.

## The claims, written before the measurements

Everything so far is a count or a simulation, and the decision to keep the 8-bit block needs claims that a measurement could refute. Chapter 4's rule is that the claim, its margin, the measurement and the decision rule are fixed before the run, so that the run can only confirm or refute them. Two claims are written down here.

The first is about work. **Claim 1:** with the feed-forward weights in 8-bit codes with a scale per 32 weights, a decoding step of the 8-layer decoder reads at least 1.4 times fewer bytes than with 32-bit weights, at every context up to 512 cached positions; the measurement is the whole step through the simulated cache, and the claim is refuted if its ratio at a context of 512 is below 1.4. The cache-trap table gives 1.5155 and the claim stands. The count says more than the claim: the ratio falls through 1.4 between contexts of 975 and 976, so the same claim stated for "every context" would have been false at 976, and the scope of the claim is part of the claim. A claim of the block's 3.55 for the whole step, or of the microbenchmark's 35.5, would have been refuted by the first whole-step measurement.

The second is about the numbers. **Claim 2:** the 8-bit block's branch $\mathbf{f}$ differs from the 32-bit block's by a relative error below 2 percent; the measurement is the relative error over eight independent draws of the weights and inputs, and the claim is refuted if the upper end of its 95 percent interval reaches 2 percent. The margin comes from an error model, not from the run, and the model is approximate: it treats the rounding errors as independent, adds them in quadrature and ignores how the SiLU and the product with the up projection reshape them. A weight rounded to a step $s$ is off by an error spread evenly over $[-s/2, s/2]$, whose root mean square is $s/\sqrt{12}$; for weights spread evenly over $[-a, a]$ the step is $a/127$ and their root mean square is $a/\sqrt{3}$, so each matrix carries a relative error of about $1/(2 \cdot 127) = 0.39$ percent, and three matrices with independent errors about $\sqrt{3} \times 0.39 = 0.68$ percent. The margin allows three times that, because trained weights are not spread evenly: a row can hold a few entries far larger than the rest, and the scale of a group is set by its largest entry. The draws plant one such entry in every row, a weight multiplied by 16.

<!-- worked: project_reproduction.quality -->
The quality of 8-bit weights on the component at D = 256, F = 768, 4 positions, eight draws. Draw k uses the generator `project_reproduction.draw.k` with the seed printed in its row, and draws in this order: the gain, W_gate, W_up and W_down by the rules of the first component (with 1/sqrt(256) and 1/sqrt(768)), then, for every row of W_gate, of W_up and of W_down in that order, one integer(k) c over the row's k columns, and the weight at column c is multiplied by 16 (the outlier of the row); then the 4 x 256 inputs as 2u - 1. Weights and inputs are rounded to 32 bits; everything after is 64-bit. Each cell is the rel_l2 of the 8-bit block against the same block with the weights as drawn. 'Group of 32' has one scale per 32 consecutive weights of a row; 'one scale per row' has one for the whole row; the column without outliers redraws the same weights and skips the multiplication (the integers are still drawn); the last column measures the block output y, residual included, where the others measure f, the down projection. The interval is the mean plus or minus t s / sqrt(8) with s the sample standard deviation (divisor 7) and t = 2.36462, the 0.975 quantile of Student's t with 7 degrees of freedom.

| draw | seed | f, group of 32 | f, one scale per row | f, one per row, no outliers | y, group of 32 |
|---|---:|---:|---:|---:|---:|
| 0 | 14186093670461477176 | 1.51e-02 | 5.05e-02 | 6.65e-03 | 6.58e-03 |
| 1 | 3588567796689332234 | 1.50e-02 | 5.08e-02 | 6.70e-03 | 6.33e-03 |
| 2 | 8114807110829101766 | 1.61e-02 | 4.87e-02 | 7.22e-03 | 6.20e-03 |
| 3 | 9599345445007377170 | 1.46e-02 | 5.08e-02 | 6.81e-03 | 5.91e-03 |
| 4 | 7194242738548454826 | 1.47e-02 | 5.12e-02 | 6.81e-03 | 5.92e-03 |
| 5 | 13413291850499546480 | 1.51e-02 | 4.59e-02 | 6.91e-03 | 6.60e-03 |
| 6 | 18416395511386013294 | 1.48e-02 | 4.61e-02 | 6.45e-03 | 6.29e-03 |
| 7 | 2753681920705353910 | 1.39e-02 | 4.55e-02 | 6.42e-03 | 5.83e-03 |
| mean |  | 1.49e-02 | 4.87e-02 | 6.75e-03 | 6.21e-03 |
| sample standard deviation |  | 6.17e-04 | 2.49e-03 | 2.59e-04 | 2.99e-04 |
| 95% interval, lower end |  | 1.44e-02 | 4.66e-02 | 6.53e-03 | 5.96e-03 |
| 95% interval, upper end |  | 1.54e-02 | 5.08e-02 | 6.96e-03 | 6.46e-03 |
<!-- /worked -->

With a scale per 32 weights the error is 1.49 percent on average over the eight draws, and the upper end of the interval, 1.54 percent, is below the margin: the claim stands. The other three columns are three ways to defend a different block with the same numbers.

The column without outliers is the error model's own case, and it measures 0.675 percent against the predicted 0.68; with one scale per row, those weights would pass any margin above 0.7 percent. On the draws with outliers, the same per-row scheme measures 4.87 percent and the upper end of its interval is 5.08 percent: the outlier of a row sets the step of every weight of the row, sixteen times too coarse for the rest. A test on clean weights would have confirmed a claim that the first real checkpoint refutes. The group of 32 confines the damage to the outlier's own group, which is why it costs 66,560 more bytes per block than one scale per row, and is worth them.

The last column measures the same 8-bit block at $\mathbf{y}$ instead of $\mathbf{f}$. The residual adds $\mathbf{x}$, which no change of the block can touch, to both sides of the comparison, and the relative error falls to 0.621 percent, 2.4 times smaller. The quantity was the branch the change makes; measuring it after the residual would make any block look good, and Chapter 2's scale traps are of the same family, a denominator chosen so that the error is small.

The interval is there because a single draw would also have been a choice. The best of the eight draws in the group column is 1.39 percent and the worst 1.61 percent; reporting only the best is the selection that the reporting rules of Volume II, Chapter 8 exist to prevent, and the interval states what the eight draws can and cannot support. The claim, its generators and seeds, its margin and its decision rule written beside its result are, at the scale of one table, what the reproducibility package of Chapter 4 holds for a whole report: the result becomes a function of records that anyone can run again.

Your claims are reproduced when your eight draws give the four columns of the table to the digits printed, and your interval the four summary rows.

## A gate that holds what was shown

Every check above was run once. A regression gate (Chapter 5) runs them on every change, so that what was shown stays true while the code changes around it; each of its checks must fail on the defective code, pass on the corrected code, and run in the configuration where its defect is visible, which for the misplaced constant means on the probe inputs. It has seven checks, each set out above: the golden is present, and its absence fails the gate instead of skipping it; every tap of the port is within its bound on the unit inputs and on the probe; the gradient check passes; the 8-bit block agrees with the 64-bit block that uses the dequantized weights $s\,q$, within $(2D + F + 12)\,2^{-53}$ (the count of roundings on the longest chain to $\mathbf{y}$ in 64-bit arithmetic, $D$ for the norm's sum of squares, $D$ for the gate or up projection, $F$ for the down projection, plus 12 for the norm's other steps, the SiLU, the product and the residual; the grouped sums of the 8-bit block have chains no longer than the plain ones, so the count is a bound, and the measured error under the table sits far below it); the upper end of the quality interval is below 2 percent; and the counted bytes of one decoding step at a context of 512 equal 26,314,752 exactly. The last check is a gate on work, not on time: a count is the same on every machine and every run, so it can be held to equality, and a change that alters it, for better or worse, must change the expected number and say why.

Seven regressions are run against it, each a change a reasonable person might make later, and each against a weaker gate that is common in practice: one that skips, and reports a pass, when its golden file is missing, and otherwise passes when the cosine of $\mathbf{y}$ with the golden is within $10^{-4}$ of 1.

<!-- worked: project_reproduction.gate_table -->
Seven checks of the gate, and a weak gate beside them, against seven regressions, each made alone on the specified system. golden: the golden file is present (its absence fails, it never skips); unit and probe: every tap of the 32-bit port within its bound on the two sets of inputs (not run without a golden); grad: the gradient check below 1e-6; fast: the 8-bit block of draw 0 against the 64-bit block with the dequantized weights s q, rel_l2 of f at most (2D + F + 12) 2^-53 = 1.43e-13 with D = 256, F = 768; quality: the upper end of the 95% interval of the quality table below 0.02; bytes: the counted bytes of one decoding step at context 512 equal to 26,314,752 exactly (the copy of the full-width regression is counted as 4 bytes written and 4 bytes read per weight). The weak gate skips (and reports a pass) when the golden is missing, and otherwise passes when 1 - cosine of y on the unit inputs is at most 1e-4.

| regression | defect class | golden | unit | probe | grad | fast | quality | bytes | weak gate |
|---|---|---|---|---|---|---|---|---|---|
| none (the specified system) |  | pass | pass | pass | pass | pass | pass | pass | pass |
| codes dequantized into a full-width copy of each matrix | work | pass | pass | pass | pass | pass | pass | FAIL | pass |
| gate and up cut in the wrong order | layout | pass | FAIL | FAIL | pass | pass | pass | pass | FAIL |
| eps added after the square root | constant | pass | pass | FAIL | pass | pass | pass | pass | pass |
| mean of the squares replaced by their sum | scale | pass | FAIL | FAIL | pass | pass | pass | pass | FAIL |
| one scale per row | granularity | pass | pass | pass | pass | pass | FAIL | FAIL | pass |
| SiLU slope taken as sigmoid(gate) | derivative | pass | pass | pass | FAIL | pass | pass | pass | pass |
| golden file missing | evidence | FAIL | not run | not run | pass | pass | pass | pass | pass |

The fast check of draw 0 measures 1.32e-15 against its bound of 1.43e-13. Counted per decoding step at context 512: the full-width copy adds 37,748,736 bytes to the specified 26,314,752 (18,874,368 written and 18,874,368 read back), while the 8-bit codes save 13,565,952 bytes against the 32-bit weights' 39,880,704.
<!-- /worked -->

The specified gate fails every regression and passes the specified system; each regression fails a check that names its defect class. The weak gate passes five of the seven. Two of them deserve a second look. The change that dequantizes the codes into a full-width copy of each matrix before multiplying (a buffer of 4 bytes per weight, by the rule of the work table) is a simplification of the kernel, and it computes the very numbers of the 64-bit reference with the dequantized weights; every comparison of numbers passes it, and only the count of bytes sees that it writes the copy and reads it back, 4 bytes each way per weight, which per decoding step adds more than twice what the 8-bit codes save against the 32-bit weights (both figures are printed under the table). The missing golden is the gate that never runs of Chapter 2: the weak gate reports a pass with nothing compared, and so would report a pass on every other row of the table too, if their goldens were missing as well.

The root-cause analysis of Chapter 5 asks, for each regression, which check should have caught it, and the table answers it in advance: the layout and scale mistakes are caught by the unit comparison, the misplaced constant only by the probe, the derivative only by the gradient check, the granularity by the quality claim and the byte count, the work regression only by the byte count, and the missing evidence only by refusing to skip. A gate that leaves out any column leaves out the regressions only that column catches.

Your gate is right when it reproduces the table cell by cell.

## The numbers your build must match

Collected in one place, the numbers the system must reproduce are these:

- the five digests of the component and its first values, and the fourteen digests of the golden;
- the import outcomes of the five mappings, including the refusal by name at 6 x 10 and the 0.206 of the missed transpose at 8 x 8;
- the six taps of the port within their bounds on both sets of inputs, and the first tap above and the errors of each of the four planted mistakes;
- the gradient check's pass on 216 numbers and the deviations 0.194 and 0.271 of the two planted backward mistakes;
- the FLOPs and bytes of the three blocks, the four simulated byte counts of the cache trap, and the step's ratio at each context, with the crossover at 975;
- the eight draws of the quality table and its intervals;
- the gate table, cell by cell.

## Where brain does each of these

Each of these steps has a larger instance in brain, and the evidence for each is its code.

The tap-by-tap comparison is `crates/testutil/src/parity.rs`. Its `Table` records, for every stage of a parity run, the cosine, the largest absolute difference and the relative L2 error, and holds every stage both to a cosine floor and to a relative-L2 ceiling, collecting every violation so that one run names every bad stage; `Report`, the streaming form, gates on a cosine floor and refuses a reference tap that is all near zero, since such a tap compares perfectly with anything. Tolerances derived from the rounding count are `crates/testutil/src/rounding.rs`, which defines $\gamma_k = k u/(1 - k u)$, the bound $\gamma_k \sum_i |x_i|$ of a sum of $k$ terms in 32 bits, and the agreement bound of twice that for two evaluations that may each be off by it; commit `65c09d5df` (12 November 2026) added it and changed the device golden tests and a model's backend parity test, which had compared against a fixed $10^{-6}$, to derive each tolerance from the bound over its own terms (for the model, one bound per logit, from its own shapes and weights).

A test whose fixture is missing goes through `brain_testutil::skip` in `crates/testutil/src/lib.rs`, which prints a `SKIP:` line naming the reason and panics instead when the environment variable `BRAIN_REQUIRE_FIXTURES` is set, so that a run meant to prove parity cannot pass by skipping. Commit `49865fe93` (16 October 2026) made the opt-out from device tests go through one function of the same kind and added `scripts/gates/check-test-skips.sh`, a ratchet that fails on any silent skip not on its allowlist and on any allowlist entry above the real count, so that the list can only shrink.

The 8-bit weights are `crates/model/src/int8.rs`. Its group is the project's: `GROUP = 32` weights per scale, the scale of a block its largest magnitude floored at $10^{-8}$ and divided by 127, each code rounded half away from zero and clamped to $\pm 127$ (the shared routines in `crates/checkpoint/src/packed/codec/rows.rs`). It was not always so: commit `f9c3977e7` (28 August 2026) replaced one scale per output row by the group of 32 throughout, and added the test `group_wise_confines_a_row_outlier_that_whole_channel_spreads_over_the_row`, which plants one outlier of 100.0 in rows of small weights and asserts that the damage stays inside the outlier's own group. Knowledge #65 records the reason and one consequence the project repeats: the storage ratio of the grouped codes is $4/1.125$, not 4, and when the storage gate went red the ceiling was re-derived from the layout, not widened. The function `check_scale_len` refuses a checkpoint whose scale tensor has one entry per row, by name, because that old shape is a valid length that would otherwise be read as the first row's group scales.

The work counts are `crates/gpu-core/src/cost.rs` (commit `4ea3a73bf`, 29 July 2026), which maps every recorded dispatch to its FLOPs, integer operations and bytes, offline from a recorded list or online as the dispatches are submitted. Its conventions differ from the project's in one respect and agree in another: it counts a multiply-accumulate as 2 operations (the project counts the $n - 1$ additions of a sum of $n$ products exactly), and it states that its bytes are a streaming estimate of each operand read or written once, not a cache model. A kernel without a formula is reported as uncovered, never as zero. The roofs are `crates/gpu-core/src/roof.rs` (commit `73466fee0`), which measures the arithmetic roof with a chain of fused multiply-adds held in registers and the bandwidth roof with an update of buffers far larger than any cache, the same care about caches that the microbenchmark above lacked; knowledge #27 records why a percentage of a hardcoded spec-sheet peak is not a measurement, and knowledge #46 records a harness that was run several times over the same blocks of a model, timed a warm cache, and mispredicted the cold run in both directions.

The performance gate, `crates/perf/src/gate.rs`, refuses to compare a candidate with a baseline unless both drew their requests from the same workload seed and ran on the same environment, and refuses a run whose correctness check failed in either direction: a broken run neither sets a floor nor passes one. A run whose correctness check never ran is still gated, and the outcome lists it as unverified.

## Other components for the same four steps

The four steps do not depend on the component. An attention block with rotary positions adds a constant (the rotary base) that, like $\epsilon$, is invisible at one input scale and plain at another, and a key and a value that a mapping can exchange without a shape error. A convolution adds a kernel layout with four axes to transpose. Replacing the 8-bit codes by 4-bit ones halves the bytes of the codes and, if the codes keep to the symmetric range from -7 to 7, makes the step of every group about 18 times coarser (127 levels on each side of zero become 7), which the error model above predicts before the quality table is run; and fusing the gate and up projections into one pass over $\mathbf{n}$ is an optimization that changes no number at all, so its gate is equality to the bit with the unfused port.

## Check your understanding

1. The ten-call benchmark says the 8-bit block reads 35.52 times fewer bytes, and the whole step says 1.5155. A colleague proposes to report the geometric mean of the two. What is wrong with that, and which number answers what a user pays?
2. The norm that sums the squares instead of averaging them has a cosine with the golden of 1 to nine places at its own tap. Predict the relative error there from the equations alone, and say why the cosine cannot see it.
3. A reviewer suggests raising the tolerance of every tap to $10^{-6}$ to make the gate "less flaky". Which bounds of the parity table would that loosen, which would it tighten, and why is neither a reason to choose a number?
4. The dequantize-to-a-buffer regression passes every numerical check. Could any comparison of outputs have caught it? What does the byte count compare instead?
5. Claim 1 holds at a context of 512 with a ratio of 1.5155. Someone extends it to "1.4 at every context" because 512 was "typical". What should they have computed first, and what does it give?
6. The per-row quantization passes on the draws without outliers with an upper interval of 0.696 percent. Find the error in concluding that one scale per row is safe.

## Summary

If every step passes, you have a feed-forward block reproduced from a published layout, a golden of six taps on two sets of inputs that your 32-bit port matches at every tap within a tolerance derived from its roundings, a backward pass that passes a gradient check on all its numbers, an 8-bit version whose bytes, FLOPs and error you have counted and measured, two claims that were stated before their measurements and survived them within stated scopes, and a gate that fails each of seven regressions on the check that names its defect class. The planted mistakes are the ways each step produces a confident wrong number: a cut that loads without complaint, a norm whose error is invisible to cosine, a constant invisible at one scale, a derivative that no golden can see, a benchmark that measured a cache, a ratio stated outside its scope, an error measured after the residual, a scheme tested on weights without outliers, and a gate that passes when its evidence is missing.

The project establishes less than a production system needs. The component is one block of a decoder, its sizes are small enough to count by hand and to simulate, and its cache is a model with one level and a strict rule of eviction; a real device has several levels, prefetching and other traffic, which the end-to-end measurement on the device itself must face. The quality claim is about the error of a block's output and not about what a whole model of such blocks does to a task's score, which a later evaluation must measure with the same discipline. And the gate holds the claims it was given; a claim nobody wrote down is a claim nobody defends. What remains is to put components like this one together into a whole engine, which is the subject of Volume XVI.

## References

- Biao Zhang and Rico Sennrich. Root Mean Square Layer Normalization. 2019.
- Noam Shazeer. GLU Variants Improve Transformer. 2020.
- Yann N. Dauphin, Angela Fan, Michael Auli and David Grangier. Language Modeling with Gated Convolutional Networks. 2017.
- Nicholas J. Higham. Accuracy and Stability of Numerical Algorithms, second edition. 2002.
- Samuel Williams, Andrew Waterman and David Patterson. Roofline: An Insightful Visual Performance Model for Multicore Architectures. 2009.
- Karl Popper. The Logic of Scientific Discovery. 1959.

## Answers

**Recall.**

1. Collect the facts from the published references (the papers, the configuration and the weight layout), generate the goldens first with a recorded script, map the published names onto the implementation and check the mapping for coverage and shape, run a smoke test at a toy size, compare stage by stage (isolated, then chained) at tolerances derived from the arithmetic, check the gradient of every trained path, and only then optimize, under the frozen parity gate; each step begins only when the one before it has passed. The golden stores the activations between the layers because a comparison of the output alone says that something is wrong and not where; with every tap, the first tap above its bound names the first equation that is wrong.
2. $N_{\mathrm{rnd}}\,u$ with $u = 2^{-24}$ (plus one rounding if the reference is stored in 32 bits): the relative error that the roundings can explain when no stage amplifies the error of its input. A mistake whose effect is smaller than that tolerance can never fail the comparison; it must be made large by the choice of inputs or checked another way (a constant compared as a value, a second input scale).
3. The central difference $(f(x + h) - f(x - h))/(2h)$ cancels the first error term of the Taylor expansion, so its truncation error is proportional to $h^2$ instead of $h$, and a step can be found at which both truncation and round-off are small. Dividing by $\max(1, |\text{analytic}|)$ makes the deviation relative for large gradients and absolute for small ones, so that a gradient near zero, where finite-difference noise dominates, does not produce a huge relative deviation.
4. The whole gains $1/((1 - p) + p/g)$, which is below the ceiling $1/(1 - p)$ however large $g$ is. The measurement of the whole unit of work a user waits for, made the way the real use makes it, says what the user pays; a measurement of the part says what the part costs under the conditions of its own test.
5. The claim, the quantity measured, the margin, the scope (the conditions under which it is said to hold), the measurement procedure and the rule that turns the result into "refuted" or "not refuted", all before the run. A gate that skips its check reports a pass when nothing was compared, which looks exactly like a pass when everything was, and so it certifies whatever happened; having no gate at least claims nothing.

**Check your understanding.**

1. The two numbers measure different things, and a mean of them measures nothing. The ten calls measure how a block that fits in a 1 MiB cache compares with one that does not when the same weights are read again and again, which no decoding step does; the whole step reads every weight of eight layers and the head once per token. The whole step at the context of the use, 1.5155 at 512 positions, is what a user pays, and it falls with the context.
2. With the sum in place of the mean, the divisor is $\sqrt{D m + \epsilon}$ instead of $\sqrt{m + \epsilon}$ for the mean square $m$, so the output is the right output times $\sqrt{(m + \epsilon)/(Dm + \epsilon)}$, which is $1/\sqrt{D} = 0.408$ up to terms in $\epsilon/m$. The relative error is $1 - 1/\sqrt{6} = 0.591752$. The factor depends on each row's $m$ only through $\epsilon/m$, near $3 \times 10^{-6}$, so it is the same for every row to about six digits, the output is very nearly the golden times one constant, and the cosine is 1 to nine places. The measured error, 0.592 to the digits printed, is 0.5917513 to seven: the small differences between the rows' factors and the rounding of the port move only the seventh digit.
3. $10^{-6}$ is above the bounds of $\mathbf{n}$ ($7.75 \times 10^{-7}$) and below those of the gate, up, $\mathbf{h}$, $\mathbf{f}$ and $\mathbf{y}$ ($1.13 \times 10^{-6}$ to $2.09 \times 10^{-6}$), so it loosens the first and tightens the others. The measured errors are 5 to 31 times below the derived bounds, so the gate is not flaky; a number chosen to make a gate quiet is chosen from the failures it would hide, while the derived bound says what rounding alone can explain.
4. No: it computes exactly what the 64-bit reference with the dequantized weights computes, so every comparison of outputs passes. The byte count compares the work the code does with the work it was shown to do; it sees the 4-byte copy written and read for every weight of the three matrices.
5. The ratio as a function of the context, from the counting rule, before generalizing: it falls as the cache term grows and crosses 1.4 between 975 and 976 cached positions, so "every context" is refuted at 976. A claim's scope is stated with the claim and checked at its boundary.
6. The draws without outliers are the case the error model assumes, not the case real weights present. With one weight per row 16 times larger, the per-row scale sets a step 16 times too coarse for the other weights of the row, and the error rises to 4.87 percent on average, above the margin. A test passes on clean inputs and the claim fails on the first realistic ones; the inputs of a test must contain what the real ones contain.
