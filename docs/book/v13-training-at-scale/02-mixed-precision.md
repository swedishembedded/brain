<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Mixed-Precision Training

*How do we train in 16 or 8 bits without the model falling apart?*

Take a weight equal to 1, store it in bf16, and add 0.001 to it a thousand times. In exact arithmetic it ends at 2. In bf16 it ends at 1, and not one of the thousand additions changes it. Nothing overflowed, nothing was NaN, no error was raised: each addition computed the exact sum, 1.001, and rounded it to the nearest bf16 number, which is 1. The loop is as silent as it is wrong, and it is what a naive 16-bit training run does to every weight whose updates are small compared with the weight.

The remedies all come from asking where the thousand updates should have been kept. They are three, and each answers a different failure. Updates are kept in a wider copy of the weight (**master weights**). Updates that must land in a narrow number are rounded up with a probability that makes their average right (**stochastic rounding**). Gradients that are too small for a 16-bit format are multiplied by a constant on the way and divided by it on the way back (**loss scaling**). A fourth remedy, 8-bit floating point for the products of the forward and backward passes (**FP8 training**), is the same three ideas pushed to the point where the formats have 3 or 2 fraction bits. Each remedy is derived from a computation small enough to do by hand, and each is then checked against brain's code, which holds the master copy and the rounding.

The remedies lean on four results from earlier volumes. Try them before reading on, since the reasoning below uses each:

1. (Volume IV, Chapter 1, the bf16 format.) What is the next bf16 number above 1, and how small must an update to a weight of 1 be for round to nearest to lose it?
2. (Volume IV, Chapter 1, the fp8 formats.) What is the largest finite E4M3 number, and which scale $s$ puts a block whose largest magnitude is 3000 at that number?
3. (Volume III, Chapter 7, AdamW.) On the first step the bias-corrected moments are $\hat{m} = g$ and $\hat{v} = g^2$. How far does a weight move on that step when $|g| \gg \epsilon$, and does the answer depend on the size of $g$?
4. (Volume I, Chapter 15, rounding error.) Round to nearest gives a relative error of at most $u$. What is $u$ for a format with 8 significant bits?

## Updates smaller than half a gap

The opening loop deserves its arithmetic. The bf16 numbers between $2^e$ and $2^{e+1}$ are evenly spaced, with 7 fraction bits so $2^7 = 128$ of them in each such interval, and the spacing is the **gap**

$$
\mathrm{gap}(x) = 2^{\,e-7} \quad\text{for}\quad 2^e \le |x| < 2^{e+1}.
$$

Around 1 the gap is $2^{-7} = 0.0078125$. A sum $w + \Delta$ is rounded to the nearer of the two bf16 numbers around it, and a $\Delta$ below half a gap lands closer to $w$ than to its neighbour. So a weight in $[1, 2)$ ignores every update below $0.00390625$, and the relative size of what is ignored, half a gap over $|w|$, lies between $2^{-9}$ and $2^{-8}$ depending on where $w$ stands in its interval. That is the unit round-off $u = 2^{-8}$ of the format seen from the other side: an update smaller than 0.2 to 0.4 percent of the weight is not an update.

The same loop run three more ways shows what the remedies will be. The first table repeats the thousand additions of 0.001 on a weight of 1.

<!-- worked: mixed_precision.update_stream -->
A weight starts at 1 and receives 1000 updates of 0.001, each applied as w = round(w + 0.001) where the sum is formed in exact arithmetic (float64 here) and rounded once. 'moved' counts the updates that changed the stored weight. The stochastic rows come from one generator seeded 7667754899416508645 (seed_of 'mixed_precision.update_stream') that runs across the eight runs, run after run, one uniform() per update in order; run r therefore starts where run r - 1 stopped. The fp32 row adds 0.001 rounded to fp32 at every step (ties to even). The mean of the eight stochastic finals is 1.99414.

| storage and rounding | run | weight after 1000 updates | updates that moved it |
|---|---:|---:|---:|
| bf16 weight, nearest even | - | 1 | 0 |
| fp32 weight (the master) | - | 2.00005 | - |
| bf16 view of that master | - | 2 | - |
| bf16 weight, stochastic, run 0 | 0 | 2.04688 | 131 |
| bf16 weight, stochastic, run 1 | 1 | 2.03125 | 130 |
| bf16 weight, stochastic, run 2 | 2 | 2.125 | 136 |
| bf16 weight, stochastic, run 3 | 3 | 2.03125 | 130 |
| bf16 weight, stochastic, run 4 | 4 | 2.0625 | 132 |
| bf16 weight, stochastic, run 5 | 5 | 1.84375 | 108 |
| bf16 weight, stochastic, run 6 | 6 | 1.90625 | 116 |
| bf16 weight, stochastic, run 7 | 7 | 1.90625 | 116 |
<!-- /worked -->

The nearest-even row is the opening loop. The fp32 row is what a wider copy does: the thousand additions accumulate to 2.00005, where the small excess over 2 is fp32's own rounding of 0.001, and the bf16 number the forward pass would see, the rounding of that copy, is 2. The stochastic rows are the third way. Each moves the weight only now and then, 108 to 136 times in a thousand, and by a whole gap each time, and the finals scatter around 2, from 1.84375 to 2.125, with a mean over the eight runs of 1.99414. They are not exact and not meant to be, and the rule that makes them right on average is derived below.

Before the remedies, the damage should be measured on a problem with an answer. Gradient descent on $\tfrac12 (w - t)^2$ moves the weight by $\eta (w - t)$ per step, an update that shrinks as the weight approaches the target $t$. That makes a bf16 weight stop short, and the distance at which it stops can be computed. With $t = 3.1$ the weight finally stands in $[2, 4)$ where the gap is $2^{-6} = 0.015625$, and the update $\eta |w - t|$ is absorbed once it is below half of that: $|w - t| < \mathrm{gap}/(2\eta)$. With $\eta = 0.1$ the radius is $0.078125$, five gaps, or 2.5 percent of the target.

<!-- worked: mixed_precision.stall -->
The loss is (w - t)^2 / 2 with target t = 3.1, so the gradient is w - t; plain gradient descent with rate eta = 0.1 from w = 1.0, 300 steps. Every step forms w - eta * (g) in exact arithmetic (float64) and rounds once to the storage format; fp32 rounds to fp32, ties to even. The master row evaluates the gradient at the bf16 working copy round(theta), updates the fp32 master theta, and shows the working copy round(theta). The last two columns are |w - t| after step 300 for the stored weight (the working copy in the master row) and for the master itself. Gap of the bf16 numbers around 3: 0.015625; an update is absorbed when it is below half a gap, eta |w - t| < 0.0078125, that is |w - t| < 0.078125.

| rule | after step 10 | after step 30 | after step 100 | after step 300 | |w - t| at 300 | |master - t| at 300 |
|---|---:|---:|---:|---:|---:|---:|
| fp32 weight | 2.367775 | 3.010979 | 3.099944 | 3.099999 | 1.05e-06 | 1.05e-06 |
| bf16 weight, nearest even | 2.375 | 3.015625 | 3.03125 | 3.03125 | 0.0688 | 0.0688 |
| fp32 master, bf16 working copy | 2.375 | 3.015625 | 3.09375 | 3.09375 | 0.00625 | 7.68e-04 |

Stochastic rounding of the same descent, eight runs from one generator seeded 10792451604143238950 (seed_of 'mixed_precision.stall') that runs across the runs, one uniform() per step in order. 'mean of last 100' is the average weight over steps 201 to 300. The mean of the eight averages is 3.0993.

| run | w after step 300 | mean of last 100 | mean of last 100 minus t, absolute |
|---|---:|---:|---:|
| 0 | 3.109375 | 3.0977 | 0.0023 |
| 1 | 3.09375 | 3.0994 | 6.2e-04 |
| 2 | 3.09375 | 3.0969 | 0.0031 |
| 3 | 3.109375 | 3.0991 | 9.4e-04 |
| 4 | 3.109375 | 3.1012 | 0.0012 |
| 5 | 3.09375 | 3.0978 | 0.0022 |
| 6 | 3.09375 | 3.0995 | 4.7e-04 |
| 7 | 3.109375 | 3.1025 | 0.0025 |
<!-- /worked -->

The nearest-even row confirms the prediction. The weight reaches 3.03125 and stays there, $0.0688$ from the target, which is inside the radius, while the fp32 weight arrives at $10^{-6}$. This is the weight the table calls stalled, not diverged: the loss is stuck at a floor set by the format, and no run-time check notices, since the loss still looks fine to the eye. The last two rows are the remedies of the next two sections, and the figure below puts all four on one axis.

## A full-precision copy that the updates land on

The first remedy does not change the rounding at all. It changes what is rounded. The updates are accumulated into a 32-bit copy of every weight, the **master weights** $\theta$, which the optimizer reads and writes and nothing else touches. The model's own arithmetic runs on a narrow **working copy** $\theta_{\mathrm{c}} = \mathrm{rn}(\theta)$, rebuilt from the master by rounding to nearest, where $\mathrm{rn}$ is the rounding of Volume IV, Chapter 1 to the narrow format. One step reads

$$
g = \nabla L(\theta_{\mathrm{c}}), \qquad \theta \leftarrow \theta - \eta\, \mathrm{rule}(g), \qquad \theta_{\mathrm{c}} \leftarrow \mathrm{rn}(\theta).
$$

The gradient is taken at the working copy, because that is the value the forward pass used; it updates the master, which is wider than the working copy by 16 bits. This is the same reading as the straight-through gradient of quantization-aware training, and a bf16 weight with an fp32 master is the simplest case of it.

The third row of the table above is this rule on the weight that stalled. The working copy follows the nearest-even weight at first (2.375 after 10 steps, 3.015625 after 30), and then keeps going where the other stops: it ends at 3.09375, the bf16 number nearest to 3.1 (3.09375 is $0.00625$ away, its upper neighbour 3.109375 is $0.009375$ away), while the master has crept to within $7.68 \times 10^{-4}$ of the target. The residual $0.00625$ of the working copy is not lost training. It is the rounding of the target itself to the working format, which no storage rule in 16 bits can undo, and it is about a tenth of what the weight without a master left on the table ($0.0688$).

The price is memory, and Volume IV, Chapter 1 counted it: a bf16 value, a bf16 gradient, two fp32 moments and an fp32 master cost $2 + 2 + 8 + 4 = 16$ bytes per parameter, the same as the all-fp32 layout. The saving of a master-weights run is therefore not in the parameter bytes. It is in the activations, which are held narrow, and in the bandwidth and arithmetic of every matrix product, which run on 16-bit operands; the later chapter on where training memory goes shrinks the parameter bytes in other ways, by narrowing the moments and by moving them off the device.

## Rounding up with the odds of the remainder

A master costs 4 bytes a parameter that a run short of memory may not have. The second remedy keeps the weight narrow and changes the rounding. Round to nearest is a deterministic rule, and a deterministic rule must either keep an update that is below half a gap or lose it every time. **Stochastic rounding** keeps it with the odds it deserves. Let $x$ lie between the neighbouring bf16 numbers $\ell$ and $\ell + \mathrm{gap}(x)$, and let

$$
\phi = \frac{|x| - \ell}{\mathrm{gap}(x)} \in [0, 1)
$$

be the fraction of the gap that $|x|$ covers above $\ell$ (take magnitudes, so a negative number rounds by the same rule on its magnitude). Draw $\xi$ uniformly in $[0, 1)$ and round up, away from zero, when $\xi < \phi$ and down otherwise:

$$
\mathrm{sr}(x) = \begin{cases} \ell + \mathrm{gap}(x) & \xi < \phi, \\ \ell & \text{otherwise.} \end{cases}
$$

The expectation is $\ell + \phi\,\mathrm{gap}(x) = |x|$: the rounded value is right on average, whatever the size of the update, where nearest rounding has an error of up to half a gap that always points the same way. The variance is $\phi(1 - \phi)\,\mathrm{gap}(x)^2$, at most a quarter of a gap squared, and that is the price: the result is noisier than the nearest value, which is always within half a gap of $x$.

```text
function sr_round_bf16(x: real, u: real) -> real:
    require 0.0 <= u and u < 1.0
    if x == 0.0:
        return 0.0
    a = abs(x)
    gap = 2 ^ (exponent_of(a) - 7)
    lo = floor(a / gap) * gap
    phi = (a - lo) / gap
    r = lo + gap if u < phi else lo
    return r if x >= 0.0 else -r
```

The routine takes a normal number (its exponent at least $-126$); the quotient `a / gap` is exact because the gap is a power of two. Its argument `u` is the draw $\xi$ of the rule (not the unit round-off $u$ of the recall questions). The odds are easiest to see on one number. The value $x = 1.001$ lies a fraction $\phi = 0.128$ of the way up the gap between 1 and 1.0078125, so it is sent up with that probability.

![Stochastic rounding of x = 1.001 to bf16: a number line from the lower neighbour 1 to the upper neighbour 1.0078125 with x marked near the lower end, and two arrows from x to the outcomes, rounding down to 1 with probability 0.872 and rounding up to 1.0078125 with probability 0.128.](img/rounding-odds.svg)

A table measures the same odds for $x = 1.001$, which fp32 holds as 1.00100005.

<!-- worked: mixed_precision.sr_statistics -->
x = 1.001 rounded to fp32 is 1.00100005. 100000 draws u_i = uniform() from the generator seeded 5779186313586945093 (seed_of 'mixed_precision.sr_statistics'), in order, each giving one stochastic rounding of x to bf16.

| quantity | value |
|---|---:|
| lower neighbour of x | 1 |
| upper neighbour of x (gap above) | 1.0078125 (0.0078125) |
| phi = (x - lower) / gap | 0.128006 |
| nearest even rounding of x | 1 |
| share of draws rounded up (phi predicts it) | 0.1271 |
| mean of the stochastic results | 1.000993 |
| mean minus x | -6.77e-06 |
| standard error of that mean, sqrt(phi (1 - phi) / draws) gap | 8.25e-06 |
| mean of (result - x)^2 over the draws | 6.773e-06 |
| phi (1 - phi) gap^2, the predicted value | 6.813e-06 |
| (nearest - x)^2 for nearest even rounding | 1.000e-06 |
<!-- /worked -->

Nearest rounding sends $x$ to 1 on every draw, with a squared error of $10^{-6}$. Stochastic rounding sends it to 1.0078125 on $\phi = 0.128$ of the draws, 0.1271 of them here, and its mean is $x$ to within the standard error of the mean, which is $8.25 \times 10^{-6}$ for this many draws; the mean squared error of a single draw, $6.77 \times 10^{-6}$, matches the predicted $\phi(1-\phi)\,\mathrm{gap}^2$. That is more than six times the squared error of the nearest value. Stochastic rounding is a trade of a bias for a variance, and the trade pays when many roundings are summed, because noise averages and a bias accumulates.

The row of the stall table that uses it is the fourth: the same descent with a bf16 weight and no master. Over the eight runs the weight ends at 3.09375 or at 3.109375, the two bf16 numbers on each side of the target, never stuck at 3.03125, and the average of the last hundred steps of every run is within $0.0031$ of 3.1 (the mean over the eight runs is 3.0993). The nearest-even weight stopped five gaps short; the stochastic one hovers one gap from the target and is on the right side on average.

![The distance of one weight from its target over 300 steps of the same descent with four storage rules, on logarithmic axes. The fp32 weight falls to about 1e-6. The bf16 weight with round to nearest stops at 0.0688, inside the dashed line at the absorption radius 0.0781. The bf16 working copy of an fp32 master and the bf16 weight with stochastic rounding both settle near 0.006 to 0.009, one gap or less from the target.](img/stall-curves.svg)

How is the rounding implemented without a rounding unit that draws random numbers? On the bit pattern. An fp32 number has 23 fraction bits and a bf16 number has 7, so the top 16 bits of the fp32 pattern are the bf16 pattern and the low 16 bits are the remainder that rounding must decide. Truncation drops them. The rule of Volume IV, Chapter 1 adds `0x7FFF` plus the lowest kept bit and truncates, which carries into the kept bits exactly when the remainder is above half (and, on a tie, when the kept bit is odd). Stochastic rounding adds a random 16-bit integer $r$ in place of `0x7FFF` and truncates:

```text
function sr_bits(b: u32, r: u32) -> u16:
    require r < 65536
    return u16((b + r) >> 16)
```

The addition carries into the kept bits when $r + \mathrm{low} \ge 2^{16}$, where $\mathrm{low}$ is the remainder, so exactly $\mathrm{low}$ of the $65536$ values of $r$ carry. The remainder is $\mathrm{low} \cdot 2^{e-23}$ in value and the gap is $2^{16} \cdot 2^{e-23}$, so the probability of a carry, $\mathrm{low}/65536$, is $\phi$ exactly. The count can be made for all values of $r$, with the exact mean of the results:

<!-- worked: mixed_precision.sr_bit_rule -->
Each value is written to nine significant digits (the second is exactly 1 + 2^-8); it is first rounded to fp32 (ties to even), then written as its 32-bit pattern. All 65536 values r = 0 .. 65535 are added to the pattern as an unsigned integer, the top 16 bits are kept as the bf16 pattern, and the result is compared with the pattern without the addition. 'carried' counts the r for which the bf16 pattern changed. 'mean equals x' compares the exact mean of the 65536 bf16 results with the fp32 value, in exact rational arithmetic.

| x | fp32 pattern | dropped 16 bits | r that carried | carry probability | mean equals x |
|---|---:|---:|---:|---:|---|
| 1.001 | 0x3F8020C5 | 0x20C5 = 8389 | 8389 | 8389/65536 = 0.128006 | yes |
| 1.00390625 | 0x3F808000 | 0x8000 = 32768 | 32768 | 32768/65536 = 0.500000 | yes |
| 1 | 0x3F800000 | 0x0000 = 0 | 0 | 0/65536 = 0.000000 | yes |
| -1.001 | 0xBF8020C5 | 0x20C5 = 8389 | 8389 | 8389/65536 = 0.128006 | yes |
| 3.1 | 0x40466666 | 0x6666 = 26214 | 26214 | 26214/65536 = 0.399994 | yes |
| 0.001 | 0x3A83126F | 0x126F = 4719 | 4719 | 4719/65536 = 0.072006 | yes |
<!-- /worked -->

The count is $\mathrm{low}$ in every row, a tie of the kind 1 + $2^{-8}$ carries for half of the $r$, a value that bf16 holds exactly (1) never carries, and the mean of the 65536 results equals the fp32 value exactly, including for a negative number, whose sign bit is untouched by the addition. The same routine in the same bits is the device's, which is why the carry matters in the next section.

## A draw that belongs to the element

A stochastic rounding needs one random number per rounded element per step, and that fixes a design question that the rule above does not answer: where does the number come from? The obvious source is a stream, the generator of Volume I, Chapter 17, asked for the next value as each element is reached. It has a quiet flaw. The value an element receives then depends on how many elements were rounded before it, which depends on the order of the work: the size of the tiles the update is cut into, the number of threads, and whether the run was stopped and resumed in the middle. Two runs of the same seed would round differently, and a run could not be reproduced after a restart.

A **counter-based** draw has none of that. Its number is a function of a key and of the element's absolute index, and of nothing else. A reference draw uses a scrambling of a 32-bit word, `mix32` (three xor-shifts and two multiplications), applied to the key combined with the index and once more:

```text
function mix32(x: u32) -> u32:
    x = x xor (x >> 16)
    x = x * 0x7FEB352D
    x = x xor (x >> 15)
    x = x * 0x846CA68B
    return x xor (x >> 16)

function counter_uniform(key: u32, i: u32) -> real:
    h = mix32(mix32(key xor i) xor 0x9E3779B9)
    return (h >> 8) / 2 ^ 24
```

The multiplications wrap modulo $2^{32}$ as the fixed-width types of the pseudocode do. Twelve values are rounded below in tiles of several sizes and orders; the table counts the elements that differ from the same rule applied in one tile, for the counter draw and for a stream.

<!-- worked: mixed_precision.counter_draws -->
Twelve values x_j = 1 + 0.001 (j + 1), j = 0 .. 11, are rounded to bf16 stochastically. The counter draw is u(j) = counter_uniform(0x1234abcd, j) (mix32 twice, as defined above); the stream draw takes the next uniform() of a generator seeded 1733272321624801833 (seed_of 'mixed_precision.counter_draws'), created afresh for each row, in the order the tiles are processed. Each row counts the elements whose rounded value differs from the same rule applied in one tile of 12 in order.

| order of work | elements that differ, counter draw | elements that differ, stream draw |
|---|---:|---:|
| one tile of 12 | 0 | 0 |
| tiles of 5, 5, 2 in order | 0 | 0 |
| tiles of 5, 5, 2 in reverse order | 0 | 4 |
| tiles of 3 in order | 0 | 0 |
<!-- /worked -->

The counter draw never differs, whatever the tiling or the order. The stream draw agrees while the tiles are visited in order, which hides the problem, and differs in 4 of the 12 elements when they are visited in the reverse order, as two threads taking turns would.

## Gradients below the floor of sixteen bits

Master weights and stochastic rounding protect the weights. The gradients have their own failure, and it belongs to the 16-bit format with 5 exponent bits, fp16. Its smallest positive number is $2^{-24} \approx 6 \times 10^{-8}$ and its largest is 65504, a range of about 12 decades that the gradients of a network do not always respect. Volume IV, Chapter 1 showed the share of a gradient distribution that fp16 stores as zero without a scale, and bf16, with the exponent field of fp32, needed none. What remains is how the scale is chosen while the run goes on, and why it is safe.

**Loss scaling** uses the linearity of the backward pass. If the loss is multiplied by a constant $S$, every gradient of every parameter is multiplied by $S$ (Volume III, Chapter 4: each vector-Jacobian product is linear in the arriving gradient). The backward pass runs on $S L$, so that its fp16 gradients are $S g$, shifted $\log_2 S$ octaves up into the range the format can hold. Before the optimizer sees them they are divided by $S$ in 32 bits. Two properties make this safe. If $S$ is a power of two, multiplying and dividing change only the exponent, so the scaling adds no rounding error of its own, and the division recovers the stored gradient exactly. And the division must happen before anything that reads a gradient's size: the clipping of Volume III, weight decay applied to a gradient, and the $\epsilon$ of Adam.

The last of these is easy to miss. Adam's update $\hat m / (\sqrt{\hat v} + \epsilon)$ is unchanged when every gradient is multiplied by a constant, except for $\epsilon$, which is then smaller in proportion. The next experiment shows the whole picture on six weights whose curvatures run from 1 to $3 \times 10^{-8}$, so their gradients $g_i = d_i (w_i - 1)$ run over seven decades. Only the storage of the gradient differs between the columns.

<!-- worked: mixed_precision.loss_scale_adam -->
Six weights with curvature d_i = 10^(-1.5 i), i = 0 .. 5, on L = sum_i d_i (w_i - 1)^2 / 2, all starting at w_i = 0, so g_i = d_i (w_i - 1); AdamW with rate 0.01, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-08, no weight decay, bias-corrected with the count of applied steps, 300 steps. Only the gradient's storage differs: the scaled gradient S g_i is rounded once to fp16 (ties to even; a value of 65520 or more is infinity), divided by S, and fed to the update in float64. A step whose scaled gradient has a non-finite entry is skipped whole. The dynamic scale starts at 2^16, halves on a skipped step and doubles after 20 applied steps in a row. The cells are |w_i - 1| after the last step. A number that comes from a chaotic run is printed with only the digits that are the same in the run and in four copies of it whose first weight is changed by +1e-12, -1e-12, +3e-12 and -3e-12: in full if all agree on it, as 'about x' with the leading digits on which all agree, or as 'unstable' when not even one digit survives; the other digits cannot be reproduced in another implementation. In a nudged run only weight 0 starts at the tiny amount instead of 0; the weights do not feed one another, so weights 1 to 5 are bit-identical in all five runs and are always printed in full, and only the cell of weight 0 can lose digits. 'In full' means three significant digits (scientific notation below 0.001), and the five values agree on them when they round to the same text and lie within half a unit of its last digit of one another: the stuck run at S = 2^16 leaves weight 0 at 1 - 1e-12, 1 or 1 + 1e-12, which all print as 1.

| i | d_i | fp32 gradients | fp16, no scale | fp16, S = 2^12 | fp16, S = 2^12, not divided back | fp16, S = 2^16 | fp16, dynamic from 2^16 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 1.83e-04 | 1.83e-04 | 1.83e-04 | 1.83e-04 | 1 | 2.31e-04 |
| 1 | 0.0316 | 1.83e-04 | 1.83e-04 | 1.83e-04 | 1.83e-04 | 1 | 2.31e-04 |
| 2 | 0.001 | 1.83e-04 | 1.83e-04 | 1.83e-04 | 1.83e-04 | 1 | 2.32e-04 |
| 3 | 3.16e-05 | 1.86e-04 | 3.93e-04 | 1.86e-04 | 1.83e-04 | 1 | 2.35e-04 |
| 4 | 1e-06 | 2.88e-04 | 0.017 | 2.87e-04 | 1.83e-04 | 1 | 3.54e-04 |
| 5 | 3.16e-08 | 0.0236 | 0.868 | 0.0236 | 1.70e-04 | 1 | 0.0252 |
| steps skipped |  | 0 | 0 | 0 | 0 | 300 | 4 |
<!-- /worked -->

Read the columns from the left. fp32 gradients end near $1.8 \times 10^{-4}$ for the first five weights, because Adam with a constant rate oscillates around the optimum, and the last weight stands at 0.0236, which is Adam's own $\epsilon = 10^{-8}$ at work: that gradient is at most $3.2 \times 10^{-8}$, of the order of $\epsilon$, so its updates are damped in fp32 as well. fp16 with no scale is the fp32 run for the first three weights, loses the fourth a little ($3.93 \times 10^{-4}$), the fifth badly (0.017) and the sixth almost altogether (0.868 away from the target). The fifth weight's gradient is at most $10^{-6}$, already in the subnormal range of fp16 (below $6.1 \times 10^{-5}$), and is stored as zero once its residual falls below 0.03; the sixth's is at most $3.2 \times 10^{-8}$, about half of the smallest fp16 number $6 \times 10^{-8}$, so it is stored as $6 \times 10^{-8}$ or as 0. A gradient stored as 0 gives an update of 0. A scale of $2^{12}$ restores all of them to the fp32 values (the third column agrees with the first to three digits). The fourth column feeds the scaled gradient to Adam without dividing back, and the last weight then ends at $1.70 \times 10^{-4}$, better than fp32: with the gradient $4096$ times larger, $\epsilon$ is $4096$ times less important, which is a different optimizer. It is an improvement here and a silent change of hyperparameters in general, which is the reason for the division.

The fifth column is the failure of a constant that is too large. At $S = 2^{16}$ the first gradient, $d_0 (w_0 - 1) = -1$ at the start, becomes $\pm 65536$ and rounds to infinity (65520 and above does), so the step is skipped, and because nothing changed the next step meets the same overflow: all 300 steps are skipped and every weight is exactly where it began. No fixed constant is right for a whole run. The gradients shrink as the loss falls, which says the scale should rise, and they can also spike, which says it should fall.

## Moving the scale as the gradients move

The rule of dynamic loss scaling is cheap. Start high. If any scaled gradient is not finite, skip the step entirely (no moment, no weight, no step counter changes) and halve the scale. If a fixed number of consecutive steps all went through, double it. The scale then climbs until it meets the ceiling set by the largest gradient, falls one octave when it overshoots, and follows the gradients down as they shrink.

```text
function scale_step(S: real, good: int, finite: bool, grow_after: int) -> (real, int, bool):
    # returns the new scale, the new count of applied steps in a row, and whether this step is applied
    require S > 0.0 and grow_after >= 1
    if not finite:
        return (S / 2, 0, false)
    good = good + 1
    if good == grow_after:
        return (S * 2, 0, true)
    return (S, good, true)
```

The last column of the previous table is this controller with $S = 2^{16}$ at the start and `grow_after` $= 20$. It applied 296 of the 300 steps and ends within the same orders as the fp32 run ($2.31 \times 10^{-4}$ to 0.0252 against $1.83 \times 10^{-4}$ to 0.0236), the 4 skipped steps being the difference.

<!-- worked: mixed_precision.scale_trace -->
The dynamic run of the previous table, listed event by event (a step is numbered from 0). The scale started at 2^16. Over the 300 steps it halved 4 times (each with a skipped step), doubled 14 times, skipped 4 steps and ended at 2^26. The first ten events:

| event | at step | scale after it | what happened |
|---:|---:|---:|---|
| 1 | 0 | 2^15 | halved, step skipped |
| 2 | 20 | 2^16 | doubled |
| 3 | 40 | 2^17 | doubled |
| 4 | 41 | 2^16 | halved, step skipped |
| 5 | 61 | 2^17 | doubled |
| 6 | 81 | 2^18 | doubled |
| 7 | 82 | 2^17 | halved, step skipped |
| 8 | 102 | 2^18 | doubled |
| 9 | 122 | 2^19 | doubled |
| 10 | 123 | 2^18 | halved, step skipped |
<!-- /worked -->

The first event is the one the fixed $2^{16}$ could not survive: the scale halves at step 0 and the step is skipped, and the next step, at $2^{15}$, goes through. The scale then doubles at steps 20 and 40, overshoots at step 41, where the first gradient is still larger than $65520/2^{17} = 0.4999$, and halves. The same dance repeats each time the scale climbs past the ceiling that the largest gradient sets, and the ceiling rises as the gradients fall: by step 300 the first gradient has fallen to $1.83 \times 10^{-4}$ (the table above) and the scale is $2^{26}$. The cost of the scheme is the skipped steps, one per overshoot, which is why the growth interval is long (the exercises vary it).

One step of the whole method puts the pieces together, and the figure shows where each number lives.

![The mixed-precision step. The fp32 master (with the optimizer's moments) is rounded down to a working copy in bf16 or fp8, the forward and backward passes run on that copy with the loss multiplied by S, the scaled gradient is divided by S in fp32 and tested for being finite, then clipped and handed to the optimizer rule, which updates the master. A non-finite gradient skips the step and halves S.](img/mixed-precision-step.svg)

For bf16 the left half of the figure is all there is: the range of the format is that of fp32, no gradient needs a scale, and the working copy and the master are the whole method. The scale controller is the part that fp16 adds, and the part that FP8 needs in a more local form.

## Eight bits for the products

A matrix product is where the arithmetic is, and a product whose operands are 8 bits wide reads half the bytes of a bf16 one for the same number of multiply-adds. **FP8 training** runs the forward and backward products on 8-bit floating-point operands, each tensor carrying a scale, accumulates the products in a wider format, and keeps everything else (the master, the moments, the sums) wide. The two formats of Volume IV, Chapter 1 split the work. A forward operand, a weight or an activation, has a bounded range and needs precision: E4M3, with 3 fraction bits, maximum 448. A gradient has a long tail and needs range: E5M2, with 2 fraction bits and maximum 57344. Each is stored as a code and one scale $s$, read as $s$ times the decoded code, where the scale puts the tensor's largest magnitude at the format's largest finite number.

The scale has to be placed somewhere, and two choices decide the quality: how many values share one scale, and which tensor's largest magnitude, its amax, sets it. The first is a matter of range, and the next table measures it on a product $Y = XW$ whose rows of $X$ span more and more decades of magnitude, as the rows of activations of different tokens can.

<!-- worked: mixed_precision.fp8_operands -->
Y = X W with X of shape [32, 64] and W of shape [64, 16]. From the generator seeded 16545152137786851729 (seed_of 'mixed_precision.fp8_operands'): the 2048 entries of X as normal() in row-major order, then the 1024 entries of W in row-major order as 0.1 * normal(). Row i of X is then multiplied by 10^(-a i / 31) for the stated number of decades a, so the rows of X span a decades of magnitude. The reference Y is formed in float64 from the unrounded matrices; the rounded operands are multiplied and summed in float64 too (a wide accumulator). The cell is the mean over the 32 rows of ||row of the rounded product - row of Y|| / ||row of Y|| (2-norm, mean divisor 32). bf16 rounds each entry to nearest even. 'E4M3, one scale per tensor' divides each matrix by s = amax / 448, rounds to E4M3 (nearest even, saturating) and multiplies back; 'E4M3, X in blocks of 16' gives X one scale per 16 consecutive entries of a row (W keeps one per tensor); 'E5M2' is the tensor rule with 57344. Errors are printed to 4 decimals.

| decades spanned by the rows of X | bf16 | E4M3, one scale per tensor | E4M3, X in blocks of 16 | E5M2, one scale per tensor |
|---|---:|---:|---:|---:|
| 0 | 0.0024 | 0.0392 | 0.0376 | 0.0824 |
| 3 | 0.0025 | 0.0383 | 0.0376 | 0.0770 |
| 6 | 0.0025 | 0.1766 | 0.0376 | 0.0794 |
| 9 | 0.0025 | 0.4606 | 0.0376 | 0.0863 |
<!-- /worked -->

The bf16 column is the floor of the 16-bit tier: a relative error of 0.0024 per row, whatever the spread, since bf16 has the range. E4M3 with one scale per tensor has 0.0392 when the rows are alike, which is the 3-bit precision showing, and 0.0383 at three decades. It then fails with the range. After scaling, E4M3 reaches from 448 down to half its smallest subnormal, $2^{-10}$: a ratio of $448 \cdot 2^{10} = 458752$, or 5.66 decades. At six decades the smallest rows lie below that and are rounded to zero, and the row error is 0.1766, at nine decades 0.4606. Giving $X$ one scale per 16 consecutive entries of a row removes the dependence: 0.0376 at every spread, since each block is scaled to the top of the format wherever it stands. E5M2 with one scale per tensor survives the spread (0.0794 at six decades, 0.0863 at nine) because its range, from 57344 down to half its smallest subnormal $2^{-17}$, covers 9.9 decades, and pays twice E4M3's error at a spread of zero (0.0824 against 0.0392).

The same trade decides the format of the gradients. A tensor whose entries span six decades below its largest, an ordinary state for gradients, is scaled in each format and rounded.

<!-- worked: mixed_precision.fp8_gradients -->
5000 stand-in gradient magnitudes g_i = 10^(-6 u_i), u_i = uniform() drawn in order from the generator seeded 1973974073949404247 (seed_of 'mixed_precision.fp8_gradients'), so they span 6 decades below the largest. Each is divided by s = amax / (largest finite number of the format), rounded once to the format (nearest even, saturating) and multiplied back. 'zero' is the share stored as 0, 'subnormal' the share stored below the smallest normal number of the format, 'mean relative error' the mean of |stored - g| / g with a zero counted as 1, and 'error of the sum' the relative error of the sum of all stored values against the sum of all g.

| format | amax maps to | zero | subnormal | mean relative error | error of the sum |
|---|---:|---:|---:|---:|---:|
| E4M3 | 448 | 5.88% | 18.52% | 0.1139 | -0.0010 |
| E5M2 | 57344 | 0.00% | 0.00% | 0.0449 | -0.0017 |
<!-- /worked -->

E4M3 stores 5.88 percent of the entries as zero and 18.52 percent in its subnormal range, and its mean relative error, counting a zero as a complete loss, is 0.1139. E5M2 loses none of them to zero and has a mean error of 0.0449, under 40 percent of E4M3's in spite of its 2 fraction bits. The error of the sum is $-0.0010$ for one and $-0.0017$ for the other, both small: what the zeros cost is the individual small gradients, not the aggregate that the large ones dominate, and an update rule that normalises per element, as Adam does, depends on the small ones. That is the reason for the split of formats: E4M3 where values are alike in size, E5M2 where they are not.

The second choice is which amax sets the scale. Finding the amax of a tensor takes a pass over it before the scale is known, and then another to quantise it; in a fused kernel that costs memory traffic. **Delayed scaling** avoids the first pass by taking the scale from the amax values of previous steps, kept in a short history, and the cost moves to the step where the data jump.

```text
function next_scale(history: list<real>, window: int, top: real) -> real:
    # history: the amax of the tensor at each earlier step, oldest first; top: 448 for E4M3
    require len(history) >= 1 and window >= 1 and top > 0.0
    start = max(0, len(history) - window)
    m = history[start]
    for i in start + 1 .. len(history):
        m = max(m, history[i])
    return m / top
```

<!-- worked: mixed_precision.delayed_scale -->
One tensor of 1000 entries per step, x = sigma_k * normal() drawn in order from the generator seeded 1099368341331773078 (seed_of 'mixed_precision.delayed_scale') for steps k = 0 .. 15; sigma_k is 1.05 times sigma_(k-1) from sigma_(-1) = 1, and at step 10 it is multiplied by a further 4. The current scale divides amax of this step's tensor by 448; the delayed scale divides the largest amax of the previous 4 steps by 448 (step 0 uses its own amax). 'clipped' counts the entries with |x| / s >= 464 (the midpoint between 448 and the next code, so a saturating conversion changes them); 'rms error' is sqrt(sum_i (x_i - q_i)^2 / sum_i x_i^2), where x_i are the original unrounded entries (the denominator uses them, not q) and q_i is x_i after dividing by the scale, rounding to E4M3 (nearest even, saturating) and multiplying back. The factor 4 at step 10 stays in sigma for every later step (1.05 times it at step 11, and so on).

| step | sigma | amax of this step | amax the delayed scale uses | clipped, delayed | rms error, delayed | clipped, current | rms error, current |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.05 | 3.809 | 3.809 | 0 | 0.0263 | 0 | 0.0263 |
| 1 | 1.103 | 3.355 | 3.809 | 0 | 0.0264 | 0 | 0.0266 |
| 2 | 1.158 | 3.647 | 3.809 | 0 | 0.0273 | 0 | 0.0266 |
| 3 | 1.216 | 3.716 | 3.809 | 0 | 0.0255 | 0 | 0.0267 |
| 4 | 1.276 | 5.123 | 3.809 | 2 | 0.0417 | 0 | 0.0251 |
| 5 | 1.34 | 4.801 | 5.123 | 0 | 0.0259 | 0 | 0.0270 |
| 6 | 1.407 | 4.948 | 5.123 | 0 | 0.0268 | 0 | 0.0271 |
| 7 | 1.477 | 4.798 | 5.123 | 0 | 0.0277 | 0 | 0.0247 |
| 8 | 1.551 | 5.508 | 5.123 | 2 | 0.0282 | 0 | 0.0261 |
| 9 | 1.629 | 4.946 | 5.508 | 0 | 0.0270 | 0 | 0.0275 |
| 10 | 6.841 | 23.58 | 5.508 | 401 | 0.4867 | 0 | 0.0265 |
| 11 | 7.183 | 23.41 | 23.58 | 0 | 0.0267 | 0 | 0.0268 |
| 12 | 7.543 | 25.66 | 23.58 | 1 | 0.0285 | 0 | 0.0265 |
| 13 | 7.92 | 31.28 | 25.66 | 2 | 0.0384 | 0 | 0.0272 |
| 14 | 8.316 | 33.01 | 31.28 | 1 | 0.0262 | 0 | 0.0260 |
| 15 | 8.731 | 28.45 | 33.01 | 0 | 0.0254 | 0 | 0.0268 |
<!-- /worked -->

While the tensor drifts slowly the delayed scale is nearly right. The scale taken from the current amax keeps the rms error between 0.0247 and 0.0275 at every step. The delayed scale does the same, within 0.0254 to 0.0285, except at steps 4, 10 and 13, where the amax had risen past the window's largest and entries clipped (errors 0.0417, 0.4867 and 0.0384). At step 10 the standard deviation of the tensor was multiplied by 4, and the delayed scale, set by amax 5.508, put 401 of the 1000 entries beyond the largest E4M3 code. They saturate, and the step's rms error is 0.4867, eighteen times the usual, where the scale from the current amax has none clipped and an error of 0.0265. The next step has caught up. A transient of this kind in an activation costs one bad step, which is the bargain: the history removes a pass over every tensor and risks the steps where the data change faster than the window.

## How brain's parameter store describes a precision

In brain the parts of the method live in three crates. `crates/paramstore` holds the description of a precision and its byte accounting, the arithmetic of the update lives beside it in `crates/optim`, and the resident low-precision weights live in `crates/model`.

`crates/paramstore/src/precision.rs` (introduced by commit `1a5d1b89b`, 20 October 2026) says in its header what the module is: "pure description and byte accounting". A parameter's precision is four independent choices: the dtype of its stored value (`ValueDtype`, f32 or bf16), the dtype the backward pass accumulates its gradient in (`GradDtype`, f32 or bf16), the format of every optimizer-state slot (`StateFormat`: f32, bf16, 8-bit blocks with one f32 scale per block, or logarithmic 8-bit blocks), and the master (`MasterPolicy`: none, bf16 or f32), together with a flag that asks for stochastic rounding "whenever a low-precision value, master or state is re-encoded". The doc comment of `MasterPolicy::None` states the pairing of master and rounding: with no master the stored value is updated directly, "stochastic rounding is then what keeps small updates alive for a low-precision value". A `PrecisionPolicy` holds an ordered list of rules (`Matcher::NameContains`, `NameSuffix`, `MinNumel`, `All`, `Any`) and a default, and `resolve` returns the precision of the first rule that matches together with the index of that rule, so that a report can name every choice. Nothing in the module picks a format from hardware or memory pressure. `ParamPrecision::bytes` charges physical bytes of every component, scales included, and its test `bytes_are_physical_and_components_independent` pins the case of an 8-bit block format that pays 4 bytes of scale for every block.

The next table is an independent computation of the same rules on a manifest of six tensors, written from the description above and not from the code: the first rule (names with "norm") keeps normalisation gains and biases in fp32, the second (at least 1024 elements) gives the large matrices a bf16 value and gradient, 8-bit moments in blocks of 64, an fp32 master and stochastic rounding.

<!-- worked: mixed_precision.precision_policy -->
Six tensors resolved by an ordered list of two rules, the first that matches deciding: rule 0 (name contains 'norm') gives fp32 value, fp32 gradient, fp32 moments and no master; rule 1 (at least 1024 elements) gives a bf16 value, a bf16 gradient, 8-bit moments in blocks of 64 elements (one byte per element plus one fp32 scale, 4 bytes, per block, the last block counted whole), an fp32 master and stochastic rounding; the default is fp32 everything with no master. AdamW has two moment slots. Bytes are physical, scales included. 'moments' is both slots together.

| tensor | elements | rule | value | gradient | moments | master | total |
|---|---:|---:|---:|---:|---:|---:|---:|
| embed.weight | 16,384,000 | rule 1 | 32,768,000 | 32,768,000 | 34,816,000 | 65,536,000 | 165,888,000 |
| blk.0.attn.weight | 1,048,576 | rule 1 | 2,097,152 | 2,097,152 | 2,228,224 | 4,194,304 | 10,616,832 |
| blk.0.attn_norm.weight | 4,096 | rule 0 | 16,384 | 16,384 | 32,768 | 0 | 65,536 |
| blk.0.mlp.weight | 2,097,152 | rule 1 | 4,194,304 | 4,194,304 | 4,456,448 | 8,388,608 | 21,233,664 |
| out_norm.weight | 512 | rule 0 | 2,048 | 2,048 | 4,096 | 0 | 8,192 |
| blk.0.attn.bias | 512 | default | 2,048 | 2,048 | 4,096 | 0 | 8,192 |
| total | 19,534,848 |  | 39,079,936 | 39,079,936 | 41,541,632 | 78,118,912 | 197,820,416 |

With every tensor in fp32 and no master the same manifest costs 16 bytes per element, 312,557,568 bytes in all; the policy above costs 197,820,416, which is 63.29% of that.
<!-- /worked -->

The normalisation gains match rule 0 even though they have 4096 elements, which would satisfy rule 1: the first match wins, and the order of the rules is part of the policy. Each of the three big matrices costs 10.125 bytes a parameter instead of 16 ($2 + 2 + 2.125 + 4$), the embedding table of 16.4 million elements alone is 165.9 megabytes of the 197.8 in total, and the whole policy costs 63.29 percent of the all-fp32 layout. Note where the saving is: the master at 4 bytes is the largest component of each big matrix, as the arithmetic of the master-weights section said it would be, and the 8-bit moments are what bring the sum down.

What the crate does not do is the update. `ParamStore` records each tensor's precision in a map and says, in the comment of that field, that an absent entry means all-fp32, "which is what the device buffers above physically are today; the streamed optimiser in `optim` is the consumer of non-default entries". The allocation function `DeviceMoments::alloc` agrees: the device optimizer "keeps an f32 value and gradient with no master copy", and a precision that asks for anything else is an error naming the parameter, apart from the format of the moments, which may be f32, bf16, or 8-bit blocks of 64. The master and the rounding are in `crates/optim`. Its codecs (`crates/optim/src/codec.rs`) implement the stochastic rounding derived above: `f32_to_bf16(x, Some(u))` takes `r = (u * 65536) as u32 & 0xFFFF`, adds it to the fp32 bits and keeps the top 16, the `sr_bits` of the pseudocode above, and the randomness comes from a function of `(key, absolute element index)` through `mix32`, with `rounding_key(seed, step, tensor, slot)` identifying the tensor and the step. The header says why, in these terms: the draw is "not from a stream - so the result does not depend on tiling, thread count or whether the run was resumed", and `mix32` is written on 32-bit words so that the WGSL kernel, which has no 64-bit integers, draws the same numbers. The test `stochastic_rounding_is_unbiased_and_tiling_independent` holds both properties: it re-encodes a value between two 8-bit codes 4000 times and asserts that the mean is within $3 \times 10^{-4}$ of the value where round to nearest is more than $2 \times 10^{-4}$ off, and it encodes the same 32 values whole and in two tiles under one key and compares the bytes. The streamed update in `crates/optim/src/streamed.rs` is the master-weights step: for each tile it decodes the master (or the value, when there is none), the gradient and the state, applies the rule, and writes the state, the master and the value back each in its own format, with a value rounded stochastically only when the policy asks for it and the value is not f32. The commit that introduced the streamed optimizer and the codecs, `b7d283e1b` (20 October 2026), carries the same header and the same test.

The cost of leaving out the master is written as a refusal. When a resident full fine-tune holds its linears in bf16 or block-scaled FP8 (`crates/model/src/resident_ft.rs`, from commit `b87dd8e0d`, 21 October 2026), `ResidentFullFt::new` returns an error if no master copy was named: "a resident bf16/fp8 weight needs a master copy (master: f32 or bf16); without one, updates below the value's spacing are lost". That is the opening loop as an error message. The weight on the device is the quantization of the master (for FP8, one scale per `block` by `block` tile, and the device's FP8 weight tier uses `FP8_BLOCK`, 128, in `crates/model/src/ops.rs`), the gradient is taken at that quantized value and applied to the master, and after each step the stored weight is rewritten from it, the reading of a straight-through gradient above. The optimizer-level gate is `encoded_values_with_a_master_train_like_fp32_and_without_one_do_not` in `crates/optim/tests/it/full_ft_state.rs`: it trains a task with values in bf16, FP8, int8 and NF4 and, for the formats whose steps the gate says need one (FP8 and NF4), asserts that the loss without a master stays more than ten times the loss with one. A neighbouring name is easy to confuse with the loss scale: `grad_scale` in `crates/optim/src/rule.rs` and the `grad_multiplier` of `StepReport` are the factor "accumulation average x clip" that multiplies every gradient before the rule, not a loss scale. The stored value and the gradient of a parameter can be f32 or bf16 only (`ValueDtype` and `GradDtype`), which is why there is no scaler to find: loss scaling is the discipline of the fp16 path, and bf16 needs none.

Which products of a training step read 8-bit operands is a question about three products of each linear layer, and the code answers it product by product. Take a layer whose weight is resident in E4M3 with 128 by 128 block scales (the resident full fine-tune above, or a frozen base held as `fp8`).

- The forward product $Y = XW^{\top}$. `Ops::matmul` sends an FP8 weight to `Ops::matmul_rows` (`crates/model/src/ops.rs`), whose plan for every tier but f32 and bf16 is a tile decode (`plan_linear` in `crates/model/src/ops/tiled.rs`), and which tries `Ops::fp8_forward` first. That function returns false, and the product falls back to decoding weight tiles to a dense scratch, unless the weight is E4M3 with `FP8_BLOCK` = 128 blocks, $K$ is a multiple of 128, the row window is aligned to whole blocks, the CUDA provider (NVIDIA's GPU platform) runs and the device has FP8 linear candidates (`Gpu::has_fp8_linear`). Among those candidates two kinds exist, told apart by their declared input (`crates/gpu-core/src/native_upgrade.rs`). The dequantize-in-register kernels (`Input::DEQUANT_BF16`) read the weight's E4M3 bytes and decode them against bf16 activations: the weight is read as stored, the activation operand is bf16. Only the `Input::FP8_NATIVE` candidates, `matmul_fp8_wgmma` and its variants (which load through the tensor memory accelerator or swap the operands), read E4M3 on both sides: they need compute capability 9.0, the NVIDIA Hopper generation (`WGMMA_MIN_CC` in `crates/kernels-cuda/src/lib.rs` and `ArchFeatures::Required`), serve more than 16 activation rows (more than 32, `DECODE_REGIME_MAX_ROWS`, for the plain `matmul_fp8_wgmma`), take activations quantised to E4M3 with one f32 scale per row per 128 elements by `quant_e4m3_128`, and fold the activation and weight scales once per 128 of $K$ into an f32 sum (the wgmma instruction is the warpgroup matrix multiply-accumulate of that generation). They were added by commit `32134efec` (29 October 2026). The first listed candidate that serves a shape is a bf16 one, and the FP8-native candidate is used only where the table measured for the device (`Gpu::prepare_fp8_linear`, run when an FP8 weight is placed) names it, so the operand format of a forward product is the outcome of a measurement on the device.
- The product with the weight for the gradient of the input, $dX = dY\,W$. `Ops::matmul_dx` and `Ops::matmul_dx_rows` send the same weight to `dx_tiled`, which decodes each tile of the weight to a bf16 or f32 scratch and runs the dense product there. Both operands are dense and wide, and $dY$ is an f32 buffer.
- The product for the gradient of the weight, $dW = dY^{\top}X$. `Ops::weight_grad` and `Ops::matmul_dw` read $X$ and $dY$ as f32 and never read the weight, and accumulate in f32 or, for a bf16 gradient, in an f32 tile added to bf16 with round to nearest even.

The answer for a training step is therefore narrow. At most the forward product, and only on a compute capability 9.0 device whose tuned table selects an FP8-native candidate, multiplies 8-bit operands on both sides; the other forward candidates read an 8-bit weight against bf16 activations, and the two backward products read none. No code path forms a gradient in E5M2: the gradient of a parameter is stored as f32 or bf16 (`GradDtype`). Frozen weights in NVFP4, a 4-bit block-scaled format, go through the same forward entry point and, on CUDA from two rows up, read E4M3 activations (`Ops::nvfp4_e4m3`, unless the activations are set to bf16); that is the same pattern with a 4-bit weight. The boundaries between layers can be stored as E4M3 with one scale per row (`SavedFormat::Fp8` in `crates/model/src/saved.rs`), and are unpacked to f32 before the next product reads them, so that is storage, not a product operand.

The repository's own records bear the date of this state. Knowledge #336, written on 20 October 2026, says that brain then had no FP8 GEMM kernel and no E4M3 or E5M2 operand path in its trainers, which was true of that day. The resident FP8 tier stores and updates weights in FP8 from the next day (commit `b87dd8e0d`), and the forward product above reads E4M3 operands from 29 October (commit `32134efec`, written for serving an FP8 checkpoint at FP8, through the dispatch that a resident training weight shares).

## Summary

A narrow number cannot hold a small update, and the three ways that training avoids the loss are three built from the same two numbers, a gap and a fraction of it. Updates are accumulated in a wide master copy and the narrow working copy is rounded from it: the bf16 weight that stalled at 3.03125 reaches the nearest bf16 number to its target, and the price is 4 bytes a parameter. Updates are rounded stochastically, up with the probability of the fraction of the gap they cover: unbiased, with the variance $\phi(1-\phi)\,\mathrm{gap}^2$, exact on the bit pattern by adding a random 16-bit integer, and reproducible when the draw is a function of the element and not of the order of the work. Gradients that fp16 would store as zero are multiplied by a power-of-two scale on the way and divided on the way back, with the scale found by doubling after a run of good steps and halving, with the step skipped, on the first overflow. FP8 training applies the same ideas to 8-bit products: E4M3 for operands, E5M2 for the long-tailed gradients, one scale per tensor or per block, with the scale taken from the amax of the present step or, to save a pass, of the last few.

Bytes were counted only in the policy table, and the master, the gradient and the moments of a 7-billion-parameter model still add up to many times the model. The next chapter takes that sum apart: where each byte of a training run lives, which of them are activations and which are state, and what can be recomputed, narrowed or moved so that a model larger than the device can still be trained.

## References

- Micikevicius, P., Narang, S., Alben, J., Diamos, G., Elsen, E., Garcia, D., Ginsburg, B., Houston, M., Kuchaiev, O., Venkatesh, G. and Wu, H. Mixed precision training, 2018.
- Micikevicius, P., Stosic, D., Burgess, N., Cornea, M., Dubey, P., Grisenthwaite, R., Ha, S., Heinecke, A., Judd, P., Kamalu, J., Mellempudi, N., Oberman, S., Shoeybi, M., Siu, M. and Wu, H. FP8 formats for deep learning, 2022.
- Gupta, S., Agrawal, A., Gopalakrishnan, K. and Narayanan, P. Deep learning with limited numerical precision, 2015.
- Kalamkar, D., Mudigere, D., Mellempudi, N. and others. A study of bfloat16 for deep learning training, 2019.

## Exercises

1. **[Calculate]** A bf16 weight stands at 0.3. Give its gap, the smallest update (in magnitude) that changes it under round to nearest, ties to even, and that smallest update as a fraction of the weight. Then do the same at 12.
2. **[Calculate]** A value $x = 1.003$ (taken as exact) is rounded to bf16 stochastically. Give $\phi$, the probability of each outcome, the mean and the variance of the result, and the squared error of round to nearest.
3. **[Calculate]** The largest gradient entry of a network is 0.5. What is the largest power-of-two loss scale for which no entry of the scaled fp16 gradient is infinity, and what does that scale make of an entry of $10^{-9}$?
4. **[Calculate]** A model of $7 \times 10^9$ parameters keeps a bf16 value, a bf16 gradient, a bf16 master with stochastic rounding and two moments in 8-bit blocks of 64. Count the bytes per parameter and the total in decimal GB, scales included.
5. **[Calculate]** The amax history of an activation is 3, 5, 4, 6 (oldest first) and the delayed scale uses a window of 4 for E4M3. What is the scale, what is the smallest magnitude of the next tensor that saturates (use 464 as the threshold of the scaled values), and what is the error of an entry of magnitude 30?
6. **[Implement]** Reproduce `mixed_precision.update_stream`: the thousand additions in bf16 with round to nearest, in fp32, and in eight stochastic runs from the one generator the block names, in the order it states.
7. **[Implement]** Reproduce `mixed_precision.stall`: the four rules on the descent of the block, including the eight stochastic runs, and print the predicted absorption radius next to the measured stall.
8. **[Implement]** Reproduce `mixed_precision.sr_bit_rule`: for each value of the table, count the 65536 values of $r$ that carry and check the exact mean against the fp32 value with rational arithmetic.
9. **[Implement]** Reproduce `mixed_precision.loss_scale_adam` and `mixed_precision.scale_trace`: the six-weight AdamW run with fp16 gradient storage and the dynamic scale, with the nudges of the stable-digits rule, and the list of scale events.
10. **[Investigate]** In `mixed_precision.scale_trace` change `grow_after` from 20 to 5 and to 100. Report the number of skipped steps and the final scale for each and say what the interval trades.
11. **[Investigate]** On the descent of `mixed_precision.stall` keep the master in bf16, and update it with round to nearest in one run and with stochastic rounding in another (one run, the generator seeded `seed_of 'mixed_precision.exercise11'`, one uniform per step). Report the error of the working copy after step 300 in each and say when a bf16 master is enough.
12. **[Investigate]** Repeat `mixed_precision.delayed_scale` with a window of 1 and of 16, and with the jump at step 10 replaced by a drop to a quarter (a factor 0.25 in place of 4). Report the clipped count and the rms error at step 10 in each of the six cases, and say why a drop is cheaper than a jump.

## Check your understanding

1. A colleague says that because bf16 has the exponent range of fp32, a bf16 run needs no master weights. Find the error, using the opening example.
2. A bf16 weight stands at 8.0 and the optimizer proposes an update of 0.01. Does the stored weight change under round to nearest? Under stochastic rounding, with what probability?
3. "Stochastic rounding is unbiased, so a thousand stochastic updates of 0.001 to a weight of 1 end at exactly 2." Find the error, and say what the table of the chapter shows instead.
4. In the loss-scaled run, the gradients are clipped to a norm of 1.0. Is it correct to clip the scaled gradient? What goes wrong, and for which scales?
5. A step with an infinite scaled gradient is skipped. What would happen to Adam's moments if the step were applied with the infinity replaced by a very large number, instead of being skipped?
6. Why does the backward pass use E5M2 and the forward pass E4M3, when E4M3 has the smaller error on every tensor of the same range?
7. A resumed run, stopped after step 500 and restarted, uses a stream draw for its stochastic rounding. Why does it not reproduce the uninterrupted run, and what two numbers must a counter draw be keyed on to repeat it exactly?
8. A delayed scale is used for a tensor that grows by a factor of four in one step. Which kind of error does it make, and which does a delayed scale not make when the tensor shrinks by the same factor?

## Key terms

- **master weights**: the full-precision copy of every weight that the optimizer reads and updates, from which the narrow working copy used by the model's arithmetic is rebuilt after each step. <!-- concept: master-weights -->
- **loss scaling**: multiplying the loss by a power-of-two constant so that the backward pass's gradients fit the range of a narrow format, then dividing the gradients by it in 32 bits before anything reads their size; dynamic loss scaling halves the constant and skips the step on an overflow and doubles it after a run of good steps. <!-- concept: loss-scaling -->
- **stochastic rounding**: rounding to one of the two neighbouring numbers with the probability that makes the result right on average, up with the probability equal to the fraction of the gap the value covers. <!-- concept: stochastic-rounding -->
- **fp8 training**: running the forward and backward products on 8-bit floating-point operands (E4M3 for weights and activations, E5M2 for gradients), each tensor with a scale, accumulated and updated in wider formats. <!-- concept: fp8-training -->

## Answers

**Recall.**

1. The bf16 gap above 1 is $2^{-7} = 0.0078125$, so the next number is 1.0078125. Round to nearest loses an update below half a gap, $0.00390625$; an update of exactly that size is a tie and goes to the even neighbour, which is 1.
2. 448. The scale is $s = 3000/448 = 6.696429$.
3. After the first step $\hat m / (\sqrt{\hat v} + \epsilon) = g / (|g| + \epsilon)$, so with $|g| \gg \epsilon$ the weight moves by $\eta\,\mathrm{sign}(g)$ and the size of $g$ does not matter (the step is $\eta$ times a number just below 1).
4. $u = 2^{-8} = 0.00390625$.

**Check your understanding.**

1. The range of the exponent field is not the issue; the 7 fraction bits are. The example adds 0.001 to 1 and the sum rounds back to 1 a thousand times, in a format whose range is that of fp32. A master (or stochastic rounding) is needed for the precision of the update, not the size of the number.
2. At 8 the gap is $2^{3-7} = 0.0625$, half of it $0.03125 > 0.01$, so the weight does not change under round to nearest. Under stochastic rounding it moves up by one gap with probability $0.01/0.0625 = 0.16$.
3. The mean of the result is right, not each run. The variance of a run is the sum of $\phi(1-\phi)\,\mathrm{gap}^2$ over the updates, and the eight finals of the first table scatter from 1.84375 to 2.125 around a mean of 1.99414.
4. No. The scaled gradient's norm is $S$ times the true norm $n$, and the clip scales it by $1/(Sn)$ whenever $Sn > 1$, which for a large $S$ is every step: the gradient that remains is $S$ times too small. The division by $S$ must come first, and an overflowed step has no norm to clip.
5. An infinity makes the moments infinite or NaN for ever. A huge finite stand-in would dominate $v$ for thousands of steps ($\beta_2 = 0.999$ forgets one part in a thousand per step) and so damp every later update of that weight. Skipping changes nothing.
6. A gradient tensor spans many decades and the error of E4M3 on it is dominated by the entries it stores as zero or subnormal (the table: 5.88 percent zero, mean error 0.1139 against 0.0449 for E5M2). The forward tensors are alike in size, where precision decides.
7. A stream is consumed in the order of the work, and a resume restarts the order (the number of draws taken before an element is different, so it receives a different number). A counter draw depends only on the key and the element's absolute index, so it needs the key, built from the seed, the step, the tensor and the slot, and the index.
8. A growth makes the values above the window's largest amax saturate (401 of 1000 entries in the table, error 0.4867). A shrink leaves the delayed scale too large, so nothing saturates, and E4M3 being a floating-point format its relative error is the same at any magnitude until the subnormal range: the error hardly changes (the exercise gives the numbers).

**Exercises.**

1. 0.3 lies in $[2^{-2}, 2^{-1})$, so the gap is $2^{-9} = 0.001953125$; the smallest update that changes it is anything above $2^{-10} = 0.0009765625$ (exactly half is a tie, to the even neighbour, so the weight stays), a fraction $0.0009765625/0.3 = 0.00326$. At 12, in $[8, 16)$, the gap is $2^{-4} = 0.0625$, the half gap $0.03125$ and the fraction $0.0026$.
2. $\phi = 0.003/0.0078125 = 0.384$. The result is 1 with probability $0.616$ and 1.0078125 with probability $0.384$; the mean is 1.003; the variance is $0.384 \times 0.616 \times 0.0078125^2 = 1.444 \times 10^{-5}$; round to nearest gives 1 and a squared error of $9 \times 10^{-6}$.
3. $65504/0.5 = 131008$, and the scaled entry must stay below 65520, so $S \le 131040$: the largest power of two is $2^{16} = 65536$ (at $2^{17}$ the entry would be 65536, infinity). An entry of $10^{-9}$ becomes $6.55 \times 10^{-5}$, a normal fp16 number.
4. $2 + 2 + 2 + 2(1 + 4/64) = 6 + 2.125 = 8.125$ bytes per parameter, so $8.125 \times 7 \times 10^9 = 56.875$ GB (decimal).
5. The scale is $6/448 = 0.013393$. A scaled value saturates at $464$, so magnitudes of $464 \times 0.013393 = 6.214$ and above clip; an entry of 30 is stored as $448 s = 6$, an error of 24, which is 80 percent of the entry.
6. The block of `update_stream` states its generator and draw order; the nearest-even row stays at 1 with 0 moved updates, the fp32 row ends at 2.00005, and the stochastic rows are the eight finals listed.
7. The radius is $\mathrm{gap}(3)/(2\eta) = 0.078125$; the nearest-even weight stops at 3.03125, $0.0688$ from the target, inside it.
8. Each count equals the dropped 16 bits of the fp32 pattern, and the exact means equal the fp32 values (row for 1.0: 0 carries).
9. Match the six columns of the table, with the nudged copies, and the event list: the scale halves at step 0, doubles at 20 and 40, halves at 41.
10. With `grow_after` = 5 the run skips 42 steps, ends at $2^{25}$ and its residuals are worse ($1.61 \times 10^{-3}$ for the first weight, 0.0462 for the last), because 258 steps were applied; with 20 it skips 4 and ends at $2^{26}$; with 100 it skips 1 and ends at $2^{17}$, with residuals of $1.94 \times 10^{-4}$ to 0.024. A short interval probes the ceiling often and pays a skipped step each time; a long one tracks the falling gradients with a delay, which costs nothing here because $2^{17}$ is already above what the gradients need.
11. A bf16 master updated by round to nearest absorbs exactly the updates the bf16 weight absorbs: it ends at 3.03125, an error of 0.06875. Updated stochastically it ends at 3.09375, an error of the working copy of 0.00625 (over steps 201 to 300 the mean of $|w - t|$ is 0.0076, and $|\bar w - t|$ for the mean $\bar w$ of the weights is $6.3 \times 10^{-4}$). A bf16 master is therefore enough only when its rounding is stochastic; with round to nearest it is the first section's weight again.
12. For the jump (factor 4) the clipped counts at step 10 are 453, 401 and 401 for the windows 1, 4 and 16, with errors 0.5281, 0.4867 and 0.4867 (the window of 1 remembers only step 9's amax, 4.946, the larger windows 5.508). For the drop (factor 0.25) nothing clips in any window and the errors are 0.0259, 0.0273 and 0.0273. A drop is cheaper because it leaves the scale too large, and a floating-point format keeps its relative precision at any magnitude until the subnormal range; a jump leaves it too small, and values beyond the largest code are lost.
