<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Quantization: Fewer Bits per Weight

*How can a model's weights be stored in four bits each and still work?*

A 7-billion-weight model in 16-bit floating point is 14 GB, and generating one token reads all of it
(Chapter 3). The speed of that decoding is set by how many bytes cross the memory bus, so the way to a
quicker model is a smaller one: the same weights in fewer bits. Four bits is a quarter of sixteen. Yet four bits can name only 16 different
numbers, and a trained weight is one of billions of values that the training run chose carefully. How can sixteen levels stand in for them?

Take the smallest case that shows the whole mechanism: eight weights, stored in 4 bits each. A trained group of weights is usually not
centred on zero, so the group here is $0.03 + 0.05\,z$ for a standard normal $z$ drawn from the book's generator. We have to choose two numbers
that say what the sixteen codes mean (a step between neighbouring levels and the code that stands for zero), turn each weight into a code, read it back,
and see what that cost. We do it twice, once with a step only (the integer format of Chapter 1) and once with a step and an offset.

<!-- worked: quantization.eight_weights -->
Eight weights w_i = round(0.03 + 0.05 z_i, 4), z_i = normal() drawn in order i = 0 .. 7 from the generator, seed 17571350666410209559 (seed_of 'quantization.eight_weights', defined on the pseudocode page). Symmetric: Delta = max|w| / 7 = 0.0874 / 7 = 0.012486, q = clamp(round(w / Delta), -7, 7), w' = Delta q. Affine: range extended to contain 0, [-0.0401, +0.0874], Delta = (max - min) / 15 = 0.008500, q0 = round(-min / Delta) = 5, q = clamp(round(w / Delta) + q0, 0, 15), w' = Delta (q - q0).

| i | w | q sym | w' sym | error sym | q affine | w' affine | error affine |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | +0.0322 | 3 | +0.03746 | +0.00526 | 9 | +0.03400 | +0.00180 |
| 1 | -0.0010 | 0 | +0.00000 | +0.00100 | 5 | +0.00000 | +0.00100 |
| 2 | +0.0221 | 2 | +0.02497 | +0.00287 | 8 | +0.02550 | +0.00340 |
| 3 | +0.0075 | 1 | +0.01249 | +0.00499 | 6 | +0.00850 | +0.00100 |
| 4 | +0.0321 | 3 | +0.03746 | +0.00536 | 9 | +0.03400 | +0.00190 |
| 5 | +0.0874 | 7 | +0.08740 | +0.00000 | 15 | +0.08500 | -0.00240 |
| 6 | -0.0358 | -3 | -0.03746 | -0.00166 | 1 | -0.03400 | +0.00180 |
| 7 | -0.0401 | -3 | -0.03746 | +0.00264 | 0 | -0.04250 | -0.00240 |

Symmetric: largest |error| 0.00536 (bound Delta/2 = 0.00624), MSE 1.252e-05, codes used 6 of 15. Affine: largest |error| 0.00340 (bound Delta/2 = 0.00425), MSE 4.396e-06, codes used 7 of 16.
<!-- /worked -->

Both rows decode to numbers close to the originals: no error exceeds the half-step bound printed under the table. The affine rule, which spends its 16 codes on the
range the group really occupies, has a step of 0.0085 against 0.0125, and a mean squared error 2.8 times smaller. This chapter makes that observation exact. It derives what the error of such a code is and how
it grows, shows what a single large weight does to a scale, and builds the group-wise and two-level formats that real checkpoints use,
down to the bytes of one block. It ends with the question of which tensors may have fewer bits than others.

Five older ideas carry the rest of this chapter: the integer code and the fp8 format of Chapter 1, rounding to a grid, the mean squared error, and the roofline account of a decode step. Bring them back from memory; the answers are at the end of the chapter.

1. (Chapter 1, number formats.) An int8 code $q = -37$ is stored with the scale $\Delta = 0.01$. What number does it stand for, and what is the largest error that
   rounding to the nearest code can make?
2. (Chapter 1.) What is the largest finite E4M3 number, and why is an 8-bit floating-point format always used with a scale per block of values?
3. (Volume I, Chapter 15, floating point.) A number is rounded to the nearest point of a grid with spacing $h$. What is the largest error, and what
   is it called when the grid is the numbers of a floating-point format?
4. (Volume II, Chapter 2, linear regression.) The errors of three predictions are $0.1$, $-0.2$ and $0.3$. What is their mean squared error, and in what units is its square root?
5. (Chapter 3, the roofline.) A model with $7 \times 10^9$ weights is stored at 2 bytes per weight. How many bytes does one token read at batch size 1, and what limits the speed of decoding?

## Affine quantization: a step and a zero

The table above did the same three things in each column. It chose a **scale** $\Delta$, the distance between neighbouring levels; it chose a **zero point**
$q_0$, the code that stands for the number 0; and it mapped each weight $w$ to the code $q$ and back to $w'$. Together, the pair $(\Delta, q_0)$ is the
**scale and zero point** of a group, and the map
$$q = \mathrm{clamp}\bigl(\mathrm{round}(w/\Delta) + q_0,\; 0,\; 2^{n_b} - 1\bigr), \qquad w' = \Delta\,(q - q_0)$$
is **affine quantization**: an affine map (a multiplication and an offset) from the real line to $n_b$-bit codes and back. Chapter 1 described the stored format;
what is new here is the choice of the pair and the study of what that choice costs. When the range is symmetric about zero the offset can be dropped by using
signed codes $-q_{\max}\ldots q_{\max}$, which is the left column; the right column is the general case.

Two rules fix the pair. The codes must cover the range of the group: with the smallest and largest weight $w_{\min}$ and $w_{\max}$, the step is the range
divided by the number of gaps, $\Delta = (w_{\max} - w_{\min})/(2^{n_b} - 1)$, so that $w_{\min}$ lands near code 0 and $w_{\max}$ near code $2^{n_b} - 1$. And zero must be exactly
a code, because zeros are common (padding, pruned weights, the output of a rectifier) and a format that cannot store them exactly turns each into a small nonzero number. That is
why the range is first stretched to contain 0 and why $q_0 = \mathrm{round}(-w_{\min}/\Delta)$ is an integer.

```text
function affine_params(w: array<real>[n], nbits: int) -> (delta: real, q0: int):
    require n > 0 and 1 <= nbits <= 16, "empty group or unsupported width"
    lo = min(min(w), 0.0)
    hi = max(max(w), 0.0)
    require hi > lo, "an all-zero group has no scale"
    delta = (hi - lo) / (2^nbits - 1)
    q0 = clamp(round_even(-lo / delta), 0, 2^nbits - 1)
    return (delta, q0)

function quantize_affine(w: array<real>[n], nbits: int, delta: real, q0: int) -> array<int>[n]:
    q = zeros(int, [n])
    for i in 0 .. n:
        q[i] = clamp(round_even(w[i] / delta) + q0, 0, 2^nbits - 1)
    return q

function dequantize_affine(q: array<int>[n], delta: real, q0: int) -> array<real>[n]:
    return [delta * (q[i] - q0) for i in 0 .. n]
```

The symmetric variant of Chapter 1 sets $q_0 = 0$ and $\Delta = \max|w|/q_{\max}$ with $q_{\max} = 2^{n_b - 1} - 1$; for a 4-bit code that wastes the code $-8$ and
leaves 15 levels, and it puts the whole of a one-sided group's range on one side of zero, which is why the symmetric column of the table above has a step 47% larger than the affine one.
The price of the zero point is paid in the product: $x\cdot w' = \Delta\,x\,(q - q_0)$ needs the term $-\Delta q_0 x$ for every weight, which the section on weight-only quantization removes.

![A histogram of 4096 weights with 0.02 standard deviation, drawn as a step line; below its top, 15 vertical blue segments mark the levels of the symmetric 4-bit code, evenly spaced from -0.077 to +0.077, and above them 16 shorter orange segments mark the levels of the affine code. Most weights lie between the central levels; the outer levels sit in the sparse tails.](img/weight-levels.svg)

The figure shows what the sixteen codes mean for a bell-shaped tensor. The levels are evenly spaced, the weights are not: the outermost codes stand in the tails where there are almost no weights, and the central ones carry nearly all of them.

<!-- worked: quantization.histogram_summary -->
4096 weights 0.02 normal() in draw order from the generator, seed 11084680763937460601 (seed_of 'quantization.histogram', defined on the pseudocode page); max |w| = 0.07682. Symmetric levels: Delta = max|w| / 7 = 0.010974, 15 levels from -0.0768 to +0.0768. Affine levels: Delta = 0.009829, q0 = 8, 16 levels from -0.0786 to +0.0688. Weights within half a step of zero: 21.6% decode to 0 under the symmetric levels.
<!-- /worked -->

Every weight between two neighbouring levels is moved to the nearer one. That move is the whole of the error, and the next section computes how large it is.

## Quantization error: the bound, the average and the decibel

Before quantizing anything we want to predict what a code will cost from its step alone, so that the number of bits can be chosen in advance. The **quantization error** of a weight is the difference between the decoded and the original number, $e = w' - w$. With the map above and a weight inside
the range of the codes, $e = \Delta\,(\mathrm{round}(t) - t)$ where $t = w/\Delta$. Rounding to the nearest integer moves a number by at most one half
(the rounding error of Volume I, Chapter 15, with a grid of spacing 1), so
$$|e| \le \Delta/2.$$
A weight at the edge of the range is no exception: the representable range is shifted from $[w_{\min}, w_{\max}]$ by at most half a step because $q_0$ is rounded, and clipping a weight there moves it
by no more than that. The bound is the largest error; the mean squared error needs the distribution of the error. Suppose the fractional part of $t$ is equally likely to
be anywhere in $[-\tfrac12, \tfrac12]$, which holds when the density of the weights barely changes over one step. Then $e$ is uniform on $[-\Delta/2, \Delta/2]$ with density $1/\Delta$, and
$$\mathrm{MSE} = \mathbb{E}[e^2] = \frac{1}{\Delta}\int_{-\Delta/2}^{\Delta/2} e^2\,de = \frac{1}{\Delta}\cdot\frac{2}{3}\Bigl(\frac{\Delta}{2}\Bigr)^3 = \frac{\Delta^2}{12}.$$
This is the **uniform-error model**. The mean squared error is a third of the square of the worst error, since the average of $e^2$ over the interval is a third of its largest value. To compare
errors across tensors of different magnitude we divide the power of the weights by the error power and take a logarithm: the **signal-to-quantization-noise ratio**,
$$\mathrm{SQNR} = 10\log_{10}\frac{\mathrm{mean}(w^2)}{\mathrm{MSE}}\ \text{decibels}.$$
A weight tensor and its error are measured by the same two numbers throughout this chapter: the MSE of Volume II and the SQNR, whose gain per extra bit we derive now. If the codes
cover $[-A, A]$ with $2^{n_b}$ levels, $\Delta = 2A/2^{n_b}$ and $\mathrm{MSE} = A^2 4^{-n_b}/3$, so $\mathrm{SQNR} = 10\log_{10}(3P/A^2) + 6.02\,n_b$ for a signal power $P$:
**each added bit quarters the error power and adds $10\log_{10} 4 = 6.02$ decibels.** We test the claim on 100,000 normal weights, with the symmetric code and a scale from the largest weight:

<!-- worked: quantization.uniform_error_model -->
100000 weights w_i = normal() drawn in order from the generator, seed 9090619445670182624 (seed_of 'quantization.uniform_error', defined on the pseudocode page) (mean w^2 = 1.0023). One scale for all of them, Delta = max|w| / q_max with q_max = 2^(n-1) - 1, q = clamp(round(w / Delta), -q_max, q_max), w' = Delta q; 'predicted' is Delta^2 / 12, 'ratio' is measured over predicted, 'worst / (Delta/2)' the largest |error| in units of the bound, SQNR = 10 log10(mean w^2 / MSE) in dB, 'gain' its change from the row above.

| n bits | levels | Delta | predicted MSE | measured MSE | ratio | worst / (Delta/2) | SQNR dB | gain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 3 | 4.22330 | 1.4864e+00 | 8.9674e-01 | 0.603 | 1.000 | 0.48 |  |
| 3 | 7 | 1.40777 | 1.6515e-01 | 1.6537e-01 | 1.001 | 1.000 | 7.83 | +7.34 |
| 4 | 15 | 0.60333 | 3.0334e-02 | 3.0342e-02 | 1.000 | 1.000 | 15.19 | +7.36 |
| 5 | 31 | 0.28155 | 6.6060e-03 | 6.5906e-03 | 0.998 | 1.000 | 21.82 | +6.63 |
| 6 | 63 | 0.13624 | 1.5467e-03 | 1.5506e-03 | 1.003 | 1.000 | 28.10 | +6.28 |
| 7 | 127 | 0.06704 | 3.7449e-04 | 3.7457e-04 | 1.000 | 1.000 | 34.27 | +6.17 |
| 8 | 255 | 0.03325 | 9.2154e-05 | 9.2110e-05 | 1.000 | 1.000 | 40.37 | +6.09 |
<!-- /worked -->

From 3 bits up the measured MSE is the prediction $\Delta^2/12$ within 0.3%, and the worst error is exactly the bound: some weight lies half a step from a level. The gain per
bit falls from 7.3 decibels towards 6.02 as the model becomes accurate. At 2 bits the model fails: with only three levels, $\Delta = 4.22$ is larger than the standard deviation of the weights, most weights round to zero
and their error is the weight itself, whose mean square (the variance) is smaller than $\Delta^2/12$; the ratio of measured to predicted is 0.603 and the SQNR is 0.48 dB. The 2-bit row marks the point at which one step is no longer small against the spread of the data, and the model's assumption (a density that is flat over one step) does not hold.

The table also shows the cost of a scale taken from the largest weight: the step $\Delta = \max|w|/q_{\max}$ is set by the single most extreme of 100,000 values (4.2 standard deviations here), not by the bulk. That is harmless for a bell-shaped
tensor and ruinous for a heavy-tailed one.

## One large weight stretches every step

One failure forces every practical format to use more than one scale per tensor. A trained weight tensor often has a few weights much larger than the rest. Build one: 16,384 weights $0.02\,z$ with $z$ normal, of which each, with probability 0.004, is replaced by one ten times
as large (the seeded flags are drawn after the normals, as the table states). We quantize to 4 bits with one scale for the whole tensor and then with one scale per 32 consecutive weights, and we
look only at the weights that are not outliers, because those carry nearly all of the tensor's information.

<!-- worked: quantization.outliers -->
16384 weights, seed 9405317910815589369 (seed_of 'quantization.heavy_tailed', defined on the pseudocode page): first z_i = normal() for i = 0 .. 16383, then u_i = uniform() for i = 0 .. 16383; w_i = 0.02 z_i, and in the 'with outliers' rows w_i = 0.2 z_i where u_i < 0.004 (58 weights; the 'no outliers' rows ignore u). 4-bit symmetric codes -7 .. 7, scale = max|group| / 7 stored as f32. The columns after the scale are computed over the weights that are not outliers (the same indices in both datasets): the share that decode to 0, their MSE and their SQNR in dB. 'Delta' is the single tensor scale.

| data | scales | Delta | non-outliers decoded as 0 | MSE of non-outliers | SQNR of non-outliers dB |
|---|---|---:|---:|---:|---:|
| no outliers | one scale per tensor | 0.01129 | 21.5% | 1.064e-05 | 15.78 |
| no outliers | one scale per 32 |  | 12.9% | 3.740e-06 | 20.32 |
| with outliers | one scale per tensor | 0.05342 | 81.7% | 2.188e-04 | 2.65 |
| with outliers | one scale per 32 |  | 15.6% | 9.482e-06 | 16.28 |
<!-- /worked -->

With one scale, the 58 outliers set $\Delta$ to 0.0534, 4.7 times the step of the same tensor without them. A typical weight has magnitude 0.02, less than half a step, so 81.7% of the ordinary weights decode as exactly zero and their SQNR
is 2.65 dB: the tensor is destroyed although only 0.35% of it was unusual. With one scale per 32 the damage is confined to the groups that hold an outlier: the SQNR of the ordinary weights is 16.28 dB, against 20.32 dB for the same
groups when no outlier is present. This is the lesson that every format after this section is built to apply: **a scale must belong to a small group, so that one large weight sets the step for 31 neighbours and not for millions.**

## Group-wise quantization

The remedy is to let a small group share a scale, and the open question is how small, since a smaller group buys accuracy and costs scale bits. **Group-wise quantization** gives each run of $G$ consecutive weights (along the axis a matrix product reduces over) its own scale. Each group is quantized by the rule of Chapter 1; the scale is stored in
a floating-point format, and the decoder multiplies.

```text
function quantize_groups(w: array<real>[n], G: int, nbits: int) -> (scales: array<f16>[n // G], q: array<int>[n]):
    require n mod G == 0, "a group would straddle the end of the tensor"
    require 2 <= nbits <= 8, "unsupported width"
    qmax = 2^(nbits - 1) - 1
    scales = zeros(f16, [n // G])
    q = zeros(int, [n])
    for g in 0 .. n // G:
        amax = max([abs(w[g * G + t]) for t in 0 .. G])
        if amax > 0.0:
            scales[g] = amax / qmax          # stored in f16: the codes use the stored value
            for t in 0 .. G:
                q[g * G + t] = clamp(round_even(w[g * G + t] / scales[g]), -qmax, qmax)
    return (scales, q)
```

A group that is entirely zero keeps the scale 0 and the codes 0. The stored bits per weight count every scale, written $\mathrm{bpw}$:
$$\mathrm{bpw} = n_b + \frac{\text{scale bits}}{G}.$$
With a 16-bit scale and $G = 32$ this is $n_b + 0.5$: a 12.5% surcharge on 4-bit codes. The table shows both sides of the trade, for 4-bit codes on the heavy-tailed weights of the last section and on 16,384 Gaussian weights.

<!-- worked: quantization.group_size_tradeoff -->
4-bit symmetric codes -7 .. 7, scale max|group| / 7 stored as f16 (16 bits), so bits per weight = 4 + 16 / G. 'heavy-tailed': the 16384 weights of the previous example with outliers (seed 9405317910815589369 (seed_of 'quantization.heavy_tailed', defined on the pseudocode page)). 'gaussian': 16384 weights 0.02 normal() in draw order from the generator, seed 16290145298982480632 (seed_of 'quantization.group_gauss', defined on the pseudocode page). 'scale overhead' is the scale bits as a share of the 4 code bits. SQNR in dB over all weights.

| G | bits per weight | bytes per weight | scale overhead | SQNR heavy-tailed dB | SQNR gaussian dB |
|---|---:|---:|---:|---:|---:|
| 16384 | 4.0010 | 0.50012 | 0.02% | 3.77 | 16.07 |
| 4096 | 4.0039 | 0.50049 | 0.10% | 4.37 | 16.40 |
| 1024 | 4.0156 | 0.50195 | 0.39% | 6.54 | 17.25 |
| 256 | 4.0625 | 0.50781 | 1.56% | 10.63 | 18.04 |
| 128 | 4.1250 | 0.51562 | 3.12% | 12.64 | 18.61 |
| 64 | 4.2500 | 0.53125 | 6.25% | 15.03 | 19.42 |
| 32 | 4.5000 | 0.56250 | 12.50% | 17.41 | 20.31 |
| 16 | 5.0000 | 0.62500 | 25.00% | 19.60 | 21.43 |
| 8 | 6.0000 | 0.75000 | 50.00% | 21.92 | 22.93 |
| 4 | 8.0000 | 1.00000 | 100.00% | 24.54 | 25.04 |
<!-- /worked -->

![Two curves of SQNR in decibels for 4-bit codes against log2 of the group size, from 2 (groups of 4) to 14 (the whole tensor of 16,384 weights). Both fall as the groups grow: the heavy-tailed weights from 24.5 dB at groups of 4 to 3.8 dB for the whole tensor, the Gaussian weights from 25.0 dB to 16.1 dB. Labels under the curves give the stored bits per weight, 6.00, 4.50 and 4.02, at groups of 8, 32 and 1024.](img/group-size-tradeoff.svg)

Read the heavy-tailed column: each halving of the group adds about 2 dB between $G = 256$ and $G = 32$ and the scale overhead doubles with it, from 1.56% to 12.5%. On Gaussian weights, which have no outliers, the gain is smaller
(18.04 to 20.31 dB over the same range), because a smaller group only tracks the local spread. Group size is therefore a price-quality choice, and the table makes the break-even visible:
at 8.0 bits per weight, 4-bit codes in groups of 4 reach 24.54 dB, while 8-bit codes with one scale for the whole tensor (8.002 bits per weight) reach 28.56 dB on the same weights. Past a point, bits are better spent on the codes than on the scales.
Real formats settle on $G$ of 16 or 32, with the scale itself in 16 bits or less.

## Weight-only quantization

Smaller storage should mean less traffic, and it can, without touching the arithmetic of the activations. In **weight-only quantization** the weights are stored as low-bit codes with their scales, and the activations stay in a higher-precision format (16 or 32 bits). The codes are decoded
on the fly, inside the matrix product, to numbers of the arithmetic format; no full decoded copy of the weights is kept in memory. This is the "store narrow, compute wide" rule of Chapter 1 applied to the weights alone, and Chapter 3 gives the
reason it pays: a decode step at batch size 1 is bandwidth-bound, so its time is bounded by the bytes of weights read, and the arithmetic is nearly free. Counting those bytes for the formats of this chapter. The rows name formats that the section on k-quantization defines in full: Q8_0 (blocks of 32 int8 codes with one f16 scale, 34 bytes), Q4_0 (blocks of 32 nibbles with one f16 scale, 18 bytes), and the k-quantization formats Q4_K, Q5_K and Q6_K (super-blocks of 256 weights with two-level scales, 144, 176 and 210 bytes); the bytes per weight in the table are those block sizes over the weights of a block:

<!-- worked: quantization.decode_traffic -->
A dense model of P = 7,000,000,000 weights read once per token (batch size 1), as in Chapter 3: bytes per token = P b, FLOPs per token = 2 P, intensity 2 / b. GB = 10^9 bytes. 'fused' is the FLOPs per weight when the scale is applied once per group of G weights (2 for the fused multiply-add of x_j q_j, plus about 3 / G per weight for the group's multiply, subtract and add, taking G = 32 except Q6_K's 16); 'dequantize first' is 3 per weight (convert and scale, then the fused multiply-add).

| stored as | bytes per weight | GB per token | bytes vs fp16 | FLOP per byte | FLOPs per weight, fused | FLOPs per weight, dequantize first |
|---|---:|---:|---:|---:|---:|---:|
| fp16 | 2.00000 | 14.000 | 1.0000 | 1.000 | 2 | 2 |
| Q8_0 | 1.06250 | 7.438 | 0.5312 | 1.882 | 2.094 | 3 |
| Q6_K | 0.82031 | 5.742 | 0.4102 | 2.438 | 2.188 | 3 |
| Q5_K | 0.68750 | 4.812 | 0.3438 | 2.909 | 2.094 | 3 |
| Q4_K | 0.56250 | 3.938 | 0.2812 | 3.556 | 2.094 | 3 |
| Q4_0 | 0.56250 | 3.938 | 0.2812 | 3.556 | 2.094 | 3 |
| brain int4, f32 scale per 32 | 0.62500 | 4.375 | 0.3125 | 3.200 | 2.094 | 3 |
<!-- /worked -->

A 4.5-bit format reads 0.5625 of the 2 bytes of fp16 per weight, a 3.56-fold reduction in bytes per token, from 14.000 GB to 3.938 GB for 7 billion weights; the arithmetic intensity of the product (2 FLOPs per weight over the bytes of one weight)
rises in step, from 1 to 3.556 FLOP per byte. That is a change in work counted, not a time: how the saved bytes show on a given machine is the measurement of Chapter 11.

The arithmetic of decoding is small because the scale does not have to be applied to every weight. For one group of $G$ weights with codes $q_j$ and activations $x_j$,
$$\sum_{j} x_j\,\Delta\,(q_j - q_0) = \Delta\Bigl(\sum_j x_j q_j - q_0 \sum_j x_j\Bigr).$$
The sum $\sum_j x_j q_j$ is the multiply-add that a product needs anyway, with the code converted to a number. The scale multiplies one result per group, and the activation sum
$\sum_j x_j$ does not depend on the row of the weight matrix at all, so it is computed once per group of activations and reused by every output row. The identity is exact in real arithmetic and agrees to rounding in floating point:

<!-- worked: quantization.factored_dot -->
One group of 32 weights w_j = round(0.05 z, 4) then 32 activations x_j = round(normal(), 4), all normal() in draw order from the generator, seed 9880860142518576437 (seed_of 'quantization.factored_dot', defined on the pseudocode page). 4-bit affine codes: Delta = 0.014680, q0 = 9. Every sum is in binary64 (double precision), added from j = 0 to 31 in order; each form is then rounded to the digits shown. The difference of the two quantized forms is a few units of the last place of a double (an implementation that adds in another order, or fuses a multiply-add, gets 0 or about 2.2e-16): the table prints only that it is below 1e-15.

| form | value |
|---|---:|
| exact: sum x_j w_j | -0.46025355 |
| dequantize first: sum x_j Delta (q_j - q0) | -0.46969247 |
| factored: Delta (sum x_j q_j - q0 sum x_j) | -0.46969247 |
| sum x_j q_j | -27.032800 |
| sum x_j (shared by every row of W) | 0.551400 |
| difference of the two quantized forms | below 1e-15 |
<!-- /worked -->

The two quantized forms agree to rounding error here (below $10^{-15}$; which last bit is wrong depends on the order of the additions), and both differ from the exact product by $-0.46969 - (-0.46025) = -0.0094$: that difference is the quantization error of the weights seen through the dot product, which the section on the layer's output explains. The cost
column of the byte table counts it: a fused implementation does about $2 + 3/G$ FLOPs per weight, against 3 for decoding each weight to a number first, and in a bandwidth-bound product the difference does not show.

For a large batch the arithmetic returns to the bottleneck (the compute-bound half of Chapter 3's roofline), the saved bytes buy nothing, and the decoding has to be cheap or its cost appears. Quantizing the activations as well,
so that the product itself runs on integers, is the next chapter's subject; this chapter's formats keep activations in floating point.

## K-quantization: scales that are themselves quantized

Small groups give small errors, but each group's scale costs bits, and the question is how to keep the first without paying the second. A step and a floor per 32 weights, each stored in 16 bits, cost 32 bits beside the 128 bits of the codes: one more bit per weight, which makes 4-bit codes a 5-bit format. The **k-quantization** formats of GGML, whose block layouts the GGUF file format carries,
store the scale of a small group in fewer bits and attach to it one higher-level number. A **super-block** of 256 weights is divided into 8 sub-blocks of 32. Each sub-block has an affine code of its own, with a step $d\,\mathrm{sc}_j$ and a floor
$d_{\min}\,\mathrm{m}_j$ built from two 6-bit integers $\mathrm{sc}_j$ and $\mathrm{m}_j$, and the whole super-block stores only two 16-bit factors $d$ and $d_{\min}$. This is Q4_K when the codes have 4 bits, and the decoding is
$$w' = d\,\mathrm{sc}_j\,q - d_{\min}\,\mathrm{m}_j \qquad \text{for weight } i \text{ with } j = \lfloor i/32 \rfloor,\ q \in 0\ldots 15.$$
The scales have scales: that is the two levels. The cost of each sub-block's two numbers falls from 32 bits to 12, and the two 16-bit factors are shared by 8 sub-blocks, so the scales cost $(8 \times 12 + 32)/256 = 0.5$ bits per weight and not 1. The bytes of the formats of this chapter, from their field lists:

<!-- worked: quantization.block_formats -->
Each format is a block of weights and the bytes of its fields (the order is the order in the file, except that the fp16 'block' is one value). Bits per weight = 8 x bytes / weights.

| format | weights per block | field bytes | bytes per block | bits per weight | bytes per weight |
|---|---:|---:|---:|---:|---:|
| fp16 | 1 | 2 | 2 | 16.0000 | 2.00000 |
| Q8_0 | 32 | 2 + 32 | 34 | 8.5000 | 1.06250 |
| Q4_0 | 32 | 2 + 16 | 18 | 4.5000 | 0.56250 |
| Q5_0 | 32 | 2 + 4 + 16 | 22 | 5.5000 | 0.68750 |
| Q4_K | 256 | 2 + 2 + 12 + 128 | 144 | 4.5000 | 0.56250 |
| Q5_K | 256 | 2 + 2 + 12 + 32 + 128 | 176 | 5.5000 | 0.68750 |
| Q6_K | 256 | 128 + 64 + 16 + 2 | 210 | 6.5625 | 0.82031 |
<!-- /worked -->

Q4_0 and Q4_K cost the same 4.5 bits per weight; Q4_K spends the same bits on a better scale structure. Q8_0 is 32 int8 codes with one f16 scale: $w' = d\,q$. Q5_0 (met again in the section on import) is Q4_0 with a fifth bit per weight kept in 4 further bytes: $2 + 4 + 16 = 22$ bytes for 32 weights, $w' = d\,(q - 16)$ with $q \in 0 \ldots 31$. Q4_0 is 32 nibbles with one f16 scale: $w' = d\,(q - 8)$, with the 16 codes
$0\ldots 15$ centred on 8 and the 32 nibbles stored as 16 bytes whose low nibbles are weights $0\ldots 15$ and whose high nibbles are weights $16\ldots 31$. Q6_K is a symmetric member of the family: 6-bit codes stored as a 4-bit plane plus a 2-bit plane, 16 sub-blocks of 16 with signed 8-bit scales,
and one f16 factor $d$, so $w' = d\,\mathrm{sc}_j\,(q - 32)$ with $q \in 0 \ldots 63$.

### One Q4_K super-block, byte by byte

Everything about Q4_K is in its 144 bytes. We build one from 256 seeded weights, whose eight sub-blocks have different spreads and different centres (so that scales and floors vary), and encode it with the rule below.

<!-- worked: quantization.q4k_inputs -->
One super-block of 256 weights, seed 8407793496796183675 (seed_of 'quantization.q4k_block', defined on the pseudocode page): for sub-block j = 0 .. 7 in order, 32 weights w = round(mu_j + sigma_j z, 4) with z = normal() drawn in order (256 draws in all), sigma_j = 0.02 (1 + j mod 3), mu_j = 0.01 (j - 3). Per sub-block: the smallest and largest weight, the floor m_j = max(0, -min) and the step D_j = (max + m_j) / 15 that would use all 16 codes.

| j | sigma_j | mu_j | min | max | floor m_j | step D_j |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.02 | -0.03 | -0.0616 | +0.0144 | 0.0616 | 0.005067 |
| 1 | 0.04 | -0.02 | -0.0753 | +0.0499 | 0.0753 | 0.008347 |
| 2 | 0.06 | -0.01 | -0.1256 | +0.0986 | 0.1256 | 0.014947 |
| 3 | 0.02 | +0.00 | -0.0399 | +0.0590 | 0.0399 | 0.006593 |
| 4 | 0.04 | +0.01 | -0.0766 | +0.1277 | 0.0766 | 0.013620 |
| 5 | 0.06 | +0.02 | -0.0459 | +0.1695 | 0.0459 | 0.014360 |
| 6 | 0.02 | +0.03 | -0.0304 | +0.0737 | 0.0304 | 0.006940 |
| 7 | 0.04 | +0.04 | -0.0593 | +0.1338 | 0.0593 | 0.012873 |
<!-- /worked -->

The encoder is the one-level rule of the previous sections applied twice. For each sub-block, take the floor $m_j = \max(0, -w_{\min})$ and the step $D_j = (w_{\max} + m_j)/15$ that would use all 16 codes (the table's last two columns). These
are real numbers and the format has room for 6 bits each, so the super-block quantizes them in turn: $d = \max_j D_j / 63$, $d_{\min} = \max_j m_j/63$, both rounded to f16 first, and $\mathrm{sc}_j = \mathrm{round}(D_j/d)$, $\mathrm{m}_j = \mathrm{round}(m_j/d_{\min})$, each in $0\ldots 63$. Then,
and only then, the codes of each sub-block are found against the stored step $d\,\mathrm{sc}_j$ and the stored floor $d_{\min}\mathrm{m}_j$.

```text
record Q4K:
    d: f16
    dmin: f16
    sc: array<int>[8]
    mm: array<int>[8]
    q: array<int>[256]

function quantize_q4k(w: array<real>[256]) -> Q4K:
    D = zeros(real, [8])
    floor = zeros(real, [8])
    for j in 0 .. 8:
        sub = w[32 * j : 32 * j + 32]
        floor[j] = max(0.0, -min(sub))
        D[j] = (max(sub) + floor[j]) / 15
    d = max(D) / 63               # a value stored into an f16 is rounded to f16 here (Volume IV, chapter 1);
    dmin = max(floor) / 63        # d and dmin below are the rounded numbers, the ones the decoder will read
    sc = zeros(int, [8])
    mm = zeros(int, [8])
    q = zeros(int, [256])
    for j in 0 .. 8:
        sc[j] = clamp(round_even(D[j] / d), 0, 63) if d > 0.0 else 0
        mm[j] = clamp(round_even(floor[j] / dmin), 0, 63) if dmin > 0.0 else 0
        step = d * sc[j]
        for i in 32 * j .. 32 * j + 32:
            q[i] = clamp(round_even((w[i] + dmin * mm[j]) / step), 0, 15) if step > 0.0 else 0
    return Q4K(d, dmin, sc, mm, q)
```

<!-- worked: quantization.q4k_scales -->
The two-level scales of the super-block. d = max D_j / 63 = 0.014947 / 63 stored in f16 as 0.00023723 (bits 0x0BC6); dmin = max m_j / 63 = 0.125600 / 63 stored as 0.00199318 (bits 0x1815). sc_j = round(D_j / d) and mm_j = round(m_j / dmin) are the 6-bit integers (0 .. 63); effective step = d sc_j, effective floor = dmin mm_j; 'largest error' is max |w' - w| in the sub-block and 'in units of step/2' divides it by half the effective step.

| j | sc_j | mm_j | effective step | effective floor | largest error | in units of step/2 |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 21 | 31 | 0.004982 | 0.06179 | 0.00244 | 0.979 |
| 1 | 35 | 38 | 0.008303 | 0.07574 | 0.00414 | 0.997 |
| 2 | 63 | 63 | 0.014945 | 0.12557 | 0.00747 | 0.999 |
| 3 | 28 | 20 | 0.006642 | 0.03986 | 0.00322 | 0.969 |
| 4 | 57 | 38 | 0.013522 | 0.07574 | 0.00664 | 0.982 |
| 5 | 61 | 23 | 0.014471 | 0.04584 | 0.00681 | 0.941 |
| 6 | 29 | 15 | 0.006880 | 0.02990 | 0.00334 | 0.971 |
| 7 | 54 | 30 | 0.012810 | 0.05980 | 0.00602 | 0.940 |
<!-- /worked -->

The stored scales are close to the ideal ones but not equal: sub-block 0 has $D_0 = 0.005067$ and stores $0.004982$. A step that is smaller than the ideal one clips the largest weight of the sub-block, and one that is larger wastes codes; because the codes
are found against the stored step, no weight is more than half a stored step from its level (the last column is at most 1 here). The alternative is to find the codes against the ideal step and decode with the stored one:

<!-- worked: quantization.q4k_reassign -->
Both rows decode with the stored scales d sc_j and dmin mm_j of the super-block above; they differ only in the scales the codes were found against.

| how the codes were chosen | MSE | SQNR dB | largest |error| |
|---|---:|---:|---:|
| codes against the stored scales (d sc_j, dmin mm_j) | 9.0587e-06 | 23.58 | 0.00747 |
| codes against the ideal step D_j and floor m_j | 9.5759e-06 | 23.34 | 0.00749 |
<!-- /worked -->

The gain from using the stored scales is small on this block (an MSE 5.4% lower) and costs nothing at decoding. It is a rule of every encoder of a two-level format: the decoder only ever sees the stored numbers, so the codes must be chosen against those.

Now the bytes. The block is 144 bytes: four fields in this order.

![One Q4_K super-block of 144 bytes in four zooms on one grid. Row 1 shows the whole block: a header of bytes 0 to 15 and 128 code bytes. Row 2 shows the 16 header bytes, one cell each: the f16 d (two bytes), the f16 dmin (two bytes) and twelve scale bytes s0 to s11. Row 3 shows the twelve scale bytes in three rows of four: bytes 4 to 7 hold sc0 to sc3 in their low 6 bits and the high two bits of sc4 to sc7 above them, bytes 8 to 11 the same for m, bytes 12 to 15 the low four bits of sc4 to sc7 and of m4 to m7. Row 4 shows the first four code bytes, each with a low-nibble weight w0 to w3 and a high-nibble weight w32 to w35.](img/q4k-super-block.svg)

<!-- worked: quantization.q4k_bytes -->
The 144 bytes of the super-block of the two previous tables (index of the first byte of each row at the left).

| bytes | contents (hexadecimal) |
|---|---|
|   0.. 15 | C6 0B 15 18 D5 E3 7F DC 9F 66 3F 54 69 7D FD E6 |
|  16.. 31 | 73 B2 E7 23 E8 8B A3 58 82 88 56 48 7A 1F F3 03 |
|  32.. 47 | 80 31 A6 0D 87 66 67 65 69 B6 94 98 B2 1B A2 2C |
|  48.. 63 | A3 77 35 6C FB 67 81 67 89 53 A7 28 57 E6 35 95 |
|  64.. 79 | 8D 08 42 B7 7B 5E 79 7F 29 49 4A 45 B0 37 C6 68 |
|  80.. 95 | 77 D3 64 20 84 30 6F 9C AA 26 C8 F5 30 25 76 18 |
|  96..111 | C9 B4 8A 53 87 67 59 26 7A 14 08 57 86 66 69 38 |
| 112..127 | 7B A6 8F 40 0A 88 86 77 8A 49 34 57 AA 3C 2A 83 |
| 128..143 | CB 69 9D 57 BA 67 66 84 F2 69 29 8A 89 9C 98 68 |

Bytes 4 to 15, the twelve bytes that hold eight 6-bit scales and eight 6-bit mins:

| byte | hex | binary | holds |
|---:|---:|---|---|
| 4 | D5 | 11010101 | sc_0 (bits 0-5), sc_4 bits 4-5 (bits 6-7) |
| 5 | E3 | 11100011 | sc_1 (bits 0-5), sc_5 bits 4-5 (bits 6-7) |
| 6 | 7F | 01111111 | sc_2 (bits 0-5), sc_6 bits 4-5 (bits 6-7) |
| 7 | DC | 11011100 | sc_3 (bits 0-5), sc_7 bits 4-5 (bits 6-7) |
| 8 | 9F | 10011111 | m_0 (bits 0-5), m_4 bits 4-5 (bits 6-7) |
| 9 | 66 | 01100110 | m_1 (bits 0-5), m_5 bits 4-5 (bits 6-7) |
| 10 | 3F | 00111111 | m_2 (bits 0-5), m_6 bits 4-5 (bits 6-7) |
| 11 | 54 | 01010100 | m_3 (bits 0-5), m_7 bits 4-5 (bits 6-7) |
| 12 | 69 | 01101001 | sc_4 bits 0-3 (low nibble), m_4 bits 0-3 (high nibble) |
| 13 | 7D | 01111101 | sc_5 bits 0-3 (low nibble), m_5 bits 0-3 (high nibble) |
| 14 | FD | 11111101 | sc_6 bits 0-3 (low nibble), m_6 bits 0-3 (high nibble) |
| 15 | E6 | 11100110 | sc_7 bits 0-3 (low nibble), m_7 bits 0-3 (high nibble) |

Unpacking the twelve bytes with the rule of the chapter returns the pairs [(21, 31), (35, 38), (63, 63), (28, 20), (57, 38), (61, 23), (29, 15), (54, 30)]; they equal (sc_j, mm_j): True.

The 128 code bytes, in four chunks of 32 bytes:

| bytes | low nibbles: weights | sub-block | high nibbles: weights | sub-block |
|---|---:|---:|---:|---:|
| 16..47 | 0..31 | 0 | 32..63 | 1 |
| 48..79 | 64..95 | 2 | 96..127 | 3 |
| 80..111 | 128..159 | 4 | 160..191 | 5 |
| 112..143 | 192..223 | 6 | 224..255 | 7 |

The first eight bytes of the first chunk:

| byte | hex | low nibble q | weight | high nibble q | weight |
|---:|---:|---:|---:|---:|---:|
| 16 | 73 | 3 | 0 | 7 | 32 |
| 17 | B2 | 2 | 1 | 11 | 33 |
| 18 | E7 | 7 | 2 | 14 | 34 |
| 19 | 23 | 3 | 3 | 2 | 35 |
| 20 | E8 | 8 | 4 | 14 | 36 |
| 21 | 8B | 11 | 5 | 8 | 37 |
| 22 | A3 | 3 | 6 | 10 | 38 |
| 23 | 58 | 8 | 7 | 5 | 39 |
<!-- /worked -->

The header bytes `C6 0B` and `15 18` are the f16 values $d$ and $d_{\min}$ with the low byte first: $\mathtt{0x0BC6}$ and $\mathtt{0x1815}$. The twelve bytes after them hold sixteen 6-bit numbers, $16 \times 6 = 96$ bits, in a packing that looks irregular until one sees its rule. The first eight scales and
mins have a byte to themselves: bytes 4 to 7 hold $\mathrm{sc}_0\ldots\mathrm{sc}_3$ in their low six bits and bytes 8 to 11 hold $\mathrm{m}_0\ldots\mathrm{m}_3$. The remaining four scales and four mins ($j = 4\ldots 7$) do not fit and are cut into pieces: their two
top bits ride in the free upper two bits of the byte of $j - 4$, and their four low bits share one byte each (scale in the low nibble, min in the high nibble). The rule, for a block's 12 bytes $s_0\ldots s_{11}$ (bytes 4 to 15 of the block):

```text
function scale_min(j: int, s: array<u8>[12]) -> (int, int):
    require 0 <= j < 8, "a super-block has eight sub-blocks"
    if j < 4:
        return (s[j] and 63, s[j + 4] and 63)
    sc = (s[j + 4] and 15) or ((s[j - 4] >> 6) << 4)
    m = (s[j + 4] >> 4) or ((s[j] >> 6) << 4)
    return (sc, m)
```

Check it on byte 12 of the table, `69`: for $j = 4$ the low nibble 9 is the low four bits of $\mathrm{sc}_4 = 57 = 111001_2$ and the byte of $j - 4 = 0$, `D5` $= 11\,010101_2$, supplies the top two bits `11`: $9 + 3 \cdot 16 = 57$. The code bytes come next. 128 bytes hold 256 nibbles in four chunks of 32 bytes, and the order is not the order of the weights: in a chunk, the 32 low nibbles are the weights of one sub-block
and the 32 high nibbles the weights of the next. This is the nibble order that a reader of the format has to get right, since reading the nibbles in pairs, as a byte-sequential layout would suggest, interleaves two sub-blocks.

```text
function dequantize_q4k(b: array<u8>[144]) -> array<real>[256]:
    d = f16_of(u16(b[0]) or (u16(b[1]) << 8))
    dmin = f16_of(u16(b[2]) or (u16(b[3]) << 8))
    out = zeros(real, [256])
    for c in 0 .. 4:                                  # chunk c: bytes 16 + 32c .. 16 + 32c + 31
        (sc1, m1) = scale_min(2 * c, b[4 : 16])
        (sc2, m2) = scale_min(2 * c + 1, b[4 : 16])
        for l in 0 .. 32:
            byte = b[16 + 32 * c + l]
            out[64 * c + l] = d * sc1 * (byte and 15) - dmin * m1
            out[64 * c + 32 + l] = d * sc2 * (byte >> 4) - dmin * m2
    return out
```

Two weights, followed through by hand: byte 21 (`8B`) holds weight 5 in its low nibble and weight 37 in its high nibble.

<!-- worked: quantization.q4k_walk -->
| weight i | byte | hex | nibble | q | sub-block j | sc_j | m_j | d sc_j q - dmin m_j | decoded | original |
|---:|---:|---:|---|---:|---:|---:|---:|---|---:|---:|
| 5 | 21 | 8B | low | 11 | 0 | 21 | 31 | 0.00023723 x 21 x 11 - 0.00199318 x 31 | -0.00699 | -0.0086 |
| 37 | 21 | 8B | high | 8 | 1 | 35 | 38 | 0.00023723 x 35 x 8 - 0.00199318 x 38 | -0.00932 | -0.0069 |
<!-- /worked -->

Weight 5 is in sub-block 0 and weight 37 in sub-block 1, so the same byte is decoded with two different pairs of scales. The decoder was written from the byte layout alone; the encoder works from the fields. The two must agree, and a direct formula applied to the fields gives the check:

<!-- worked: quantization.q4k_check -->
| quantity | value |
|---|---:|
| bytes in the block | 144 |
| bits per weight | 4.5000 |
| largest |decode from bytes - direct formula| | 0.0e+00 |
| MSE of the decoded weights | 9.0587e-06 |
| SQNR dB | 23.58 |
| largest |error| | 0.00747 |
| sum of all 144 bytes | 16777 |
<!-- /worked -->

The decode from the bytes equals the direct formula $d\,\mathrm{sc}_j\,q - d_{\min}\,\mathrm{m}_j$ evaluated on the fields to the last bit, the block takes 144 bytes, 4.5 bits per weight, and its SQNR on the block's weights is 23.58 dB. In the sum of bits, $16 + 16 + 96 + 1024 = 1152 = 144 \times 8$.
The usual Q4_K encoder does not stop at the min-max step of the sketch above: it refines the steps by alternating a least-squares fit with the choice of codes, which lowers the error for the same bytes and does not change the decoder at all.

### Comparing the formats at their stored bits

All the formats of this chapter store a tensor and read it back; they differ in the bits they spend on scales. On the 16,384 heavy-tailed weights, with every scale counted:

<!-- worked: quantization.format_comparison -->
The 16384 heavy-tailed weights of the outlier example (seed 9405317910815589369 (seed_of 'quantization.heavy_tailed', defined on the pseudocode page), outliers included; max |w| = 0.3739) stored in each format. 'Q4_0': per group of 32, d = (signed element of largest magnitude) / -8 rounded to f16 (nearest, ties to even), code = clamp(round_even(w / d) + 8, 0, 15) (round_even: nearest, ties to even), w' = d (code - 8); a group of zeros decodes to zeros. 'recipe': eight sub-blocks of 32 with a 6-bit step and 6-bit floor each, an f16 d and dmin per 256 weights, as derived in the chapter (n-bit codes, 128 bits of scales per 256 weights). Bits per weight count every stored scale.

| format | bits per weight | SQNR dB | MSE | largest |error| |
|---|---:|---:|---:|---:|
| int4, one f32 scale for the tensor | 4.0020 | 3.77 | 2.190e-04 | 0.02671 |
| int4, f32 scale per 32 (brain int4) | 5.0000 | 17.41 | 9.462e-06 | 0.02639 |
| Q4_0 (f16 scale per 32) | 4.5000 | 18.52 | 7.333e-06 | 0.02279 |
| Q4_K recipe (4-bit, two-level) | 4.5000 | 20.94 | 4.203e-06 | 0.01361 |
| Q5_K recipe (5-bit, two-level) | 5.5000 | 27.22 | 9.905e-07 | 0.00678 |
| Q8_0 (f16 scale per 32) | 8.5000 | 42.36 | 3.028e-08 | 0.00147 |
| int8, one f32 scale for the tensor | 8.0020 | 28.56 | 7.259e-07 | 0.00147 |
<!-- /worked -->

![SQNR in decibels against stored bits per weight for three families on the heavy-tailed weights: one scale per tensor (about 1 dB at 3 bits, rising to 28.6 dB at 8), one symmetric f16 scale per 32 (11.5 dB at 3.5 bits to 42.4 dB at 8.5), and the two-level affine recipe (14.4 dB at 3.5 bits to 38.1 dB at 7.5). The per-tensor curve lies far below the other two over the whole range.](img/quantization-error-vs-bits.svg)

Four readings of the table and the figure. First, at 4.0 to 4.5 bits per weight the two-level recipe reaches 20.94 dB, Q4_0 18.52 dB, and 4-bit codes with one scale for the tensor 3.77 dB: 17 dB between the best and the worst arrangement of four-bit codes. Second, brain's own
4-bit tier (a symmetric $\pm 7$ code with a 32-bit scale per 32 weights, 5.0 bits) is 1.1 dB below Q4_0 at 4.5 bits: it pays 0.5 bits more for 15 levels where Q4_0 uses 16 and a half-width scale. Third, the curves of the figure rise about 6 dB per bit, the slope derived above, and
differ by an offset: how well the scales follow the data. Fourth, the per-tensor curve stays far below the others over the whole range: with 8.0 bits per weight it reaches 28.56 dB, which the 5.5-bit two-level recipe (27.22 dB) nearly matches.

## A floating-point code with a scale

Which is the better 8-bit code for a weight, an integer or a float? Chapter 1 gave the fp8 formats E4M3 and E5M2 and said they are always used with a scale. Used as a weight format they are quantization with a different set of 256 levels: instead of equally spaced levels, levels whose spacing grows with the magnitude.
Both store a scale per group (here the group maximum divided by the format's largest finite number), and the error profile differs: an integer code has an absolute error that is the same for every weight of the group, a floating-point code has a relative error that is the same
for every magnitude. We measure it on the heavy-tailed weights, grouped by 32, by the position $t = |w|/\max|\mathrm{group}|$ of each weight under its group's largest:

<!-- worked: quantization.fp8_vs_int -->
The 16384 heavy-tailed weights (seed 9405317910815589369 (seed_of 'quantization.heavy_tailed', defined on the pseudocode page)), stored in groups of 32 with one scale each: int8 (scale max|group| / 127), E4M3 (scale max|group| / 448, saturating, ties to even, with the decode of Chapter 1), E5M2 (scale max|group| / 57344) and int4 (scale max|group| / 7). Rows are bins of t = |w| / max|group| (the position of the weight under its group's largest); each cell is the mean of |w' - w| / |w| over the weights of the bin. Scales are kept exactly here, to isolate the codes.

| t of the weight | weights | int8 | E4M3 | E5M2 | int4 |
|---|---:|---:|---:|---:|---:|
| [0.5, 1] | 3860 | 0.26% | 1.96% | 3.89% | 4.72% |
| [0.25, 0.5) | 4825 | 0.55% | 2.24% | 4.45% | 8.89% |
| [0.125, 0.25) | 3441 | 1.10% | 2.27% | 4.52% | 17.87% |
| [0.0625, 0.125) | 1995 | 2.19% | 2.26% | 4.44% | 57.22% |
| [0.03125, 0.0625) | 1124 | 4.35% | 2.21% | 4.45% | 100.00% |
| [0.01562, 0.03125) | 554 | 8.62% | 2.14% | 4.62% | 100.00% |
| [0, 0.01562) | 585 | 43.82% | 2.14% | 4.64% | 100.00% |
| all weights, SQNR dB |  | 42.37 | 33.43 | 27.48 | 17.41 |
<!-- /worked -->

The integer rows grow towards the bottom as expected: int8 has a mean relative error of 0.26% for weights near the group's maximum and 43.82% for those below $1/64$ of it, where a step is as large as the weight. E4M3 stays between 1.96% and 2.27% over the same range
(the relative half-gap of a 3-bit fraction, averaged), and E5M2 between 3.9% and 4.6%, twice as large with one fraction bit fewer. Int4 loses every weight below $1/16$ of the maximum. Over the whole group, however, int8 has the best SQNR (42.37 dB against 33.43 dB for E4M3), because the mean squared error
is dominated by the largest weights, where the integer step is finer than the floating-point gap. The choice is therefore about what must be preserved. A tensor whose output is dominated by its large weights is served by the integer code; one whose small values
carry information that a large value must not drown (activations with a wide range, for example) is served by the floating-point code. Both appear in brain: weights as int8 and 4-bit groups, and activations as E4M3 (Chapter 1).

## What the layer's output sees

The error of the weights is not the one that matters; the one that does is the error of what the model computes from them. A weight error matters through the products it enters. For $y = \sum_j w_j x_j$ with weights $w'_j = w_j + e_j$, the output error is $\sum_j e_j x_j$. If the errors $e_j$ are uncorrelated with each other and with the inputs, its mean square is
$k\,\mathrm{MSE}\,\mathbb{E}[x^2]$ for $k$ inputs, while the output power is $k\,\sigma_w^2\,\mathbb{E}[x^2]$ for weights of variance $\sigma_w^2$. The $k$ and the input power cancel in the ratio:
the SQNR of the output equals the SQNR of the weights, however wide the layer. We check the prediction on a seeded layer of 64 outputs and 256 inputs, quantized group-wise at 2 to 8 bits:

<!-- worked: quantization.layer_output -->
A linear layer y = W x with W of shape [64, 256], W_ij = 0.05 normal() drawn in row-major order, then 32 inputs x with entries normal() drawn in order, one input after another, from the generator, seed 112913310546379283 (seed_of 'quantization.layer_output', defined on the pseudocode page). W is stored in n-bit symmetric codes with one f16 scale per 32 weights along each row (bits per weight = n + 0.5). 'weight SQNR' is over the weights, 'output SQNR' over all 2048 output values; 'relative output error' is the RMS error over the RMS output; cosine between the exact and the quantized outputs, both taken over the flattened vector of all 2048 outputs (input after input, output unit within it); weight and output SQNR use the mean of the squared exact values as the signal power.

| n bits | bits per weight | weight SQNR dB | output SQNR dB | difference | cosine | relative output error |
|---|---:|---:|---:|---:|---:|---:|
| 2 | 2.5 | 3.65 | 3.62 | -0.03 | 0.818277 | 65.90% |
| 3 | 3.5 | 12.95 | 13.14 | +0.19 | 0.977317 | 22.03% |
| 4 | 4.5 | 20.32 | 20.39 | +0.08 | 0.995436 | 9.56% |
| 5 | 5.5 | 26.96 | 26.96 | -0.00 | 0.998992 | 4.49% |
| 6 | 6.5 | 33.25 | 33.22 | -0.03 | 0.999762 | 2.18% |
| 7 | 7.5 | 39.35 | 39.31 | -0.04 | 0.999942 | 1.08% |
| 8 | 8.5 | 45.46 | 45.50 | +0.04 | 0.999986 | 0.53% |
<!-- /worked -->

The output SQNR tracks the weight SQNR within 0.2 dB in every row. At 4 bits plus a 16-bit scale per 32 weights the relative output error is 9.56% and the cosine similarity between exact and quantized outputs 0.995436; at 2 bits the cosine is 0.818277 and the layer is no longer the same function. Two cautions follow.
Cosine similarity ignores a change of scale (knowledge #2), so it is reported next to a norm-based quantity (here the SQNR and the relative error). And the layer here sees white inputs; real activations have a few large channels, and
when those meet the large weights the errors are not independent, so the measured output error of a real layer is the figure to trust and the decibels of this section are a prediction to check against it.

## Which tensors need how many bits

A fixed budget of bytes should go where the model is most sensitive, and not be spread as the same bits everywhere. Tensors differ. A projection that sets the scale of everything after it tolerates less error than one inside a residual branch, and an embedding table is read once per token and a layer's weights many times. Suppose each tensor $i$ has a **sensitivity** $\mathrm{sens}_i$, the increase of the loss per unit of squared weight error,
and we can store it at one of several widths $n$. The estimated loss increase of tensor $i$ at $n$ bits is then the second-order term of the loss around the trained weights,
$$E_{in} = \tfrac12\,\mathrm{sens}_i\,N_i\,\sigma_i^2\,\mathrm{rel}_n,$$
where $N_i$ is its number of weights, $\sigma_i^2$ their variance and $\mathrm{rel}_n$ the MSE over the variance of an $n$-bit code (measured below; it falls by about a factor 4 per bit at the wider widths, as derived). The bytes are $N_i(n + 0.5)/8$. The **mixed-precision weight search** chooses one width per tensor to minimize the total
$\sum_i E_{i n_i}$ within a byte budget. The problem is a multiple-choice knapsack: each tensor is a set of items of which exactly one is taken.

We make the stated model concrete. The error of one width comes from the group-wise codes of the last sections, measured once on standard normals:

<!-- worked: quantization.rel_table -->
The error of one weight at each tier: n-bit symmetric codes, f16 scale per 32 (bits per weight n + 0.5), measured as MSE over variance, the variance being the mean of w^2 (the weights have mean about 0, and it is not centred), on 65536 standard normals normal() drawn in order from the generator, seed 14869717799698414631 (seed_of 'quantization.rel', defined on the pseudocode page). This relative MSE is the rel_n of the search below.

| n bits | bits per weight | rel_n | SQNR dB |
|---|---:|---:|---:|
| 2 | 2.5 | 4.3186e-01 | 3.65 |
| 3 | 3.5 | 5.0842e-02 | 12.94 |
| 4 | 4.5 | 9.3440e-03 | 20.29 |
| 6 | 6.5 | 4.7786e-04 | 33.21 |
| 8 | 8.5 | 2.8463e-05 | 45.46 |
<!-- /worked -->

And the tensors, drawn from the book's generator:

<!-- worked: quantization.search_inputs -->
Twelve tensors, i = 0 .. 11, each drawn in order from the generator, seed 6755556699504753404 (seed_of 'quantization.tensors', defined on the pseudocode page): size N_i = 4096 (1 + integer(4)), sigma_i = 0.02 * 2^(0.5 normal()), sensitivity sens_i = 10^(2 uniform()); the draws of one tensor are made in the order integer, normal, uniform, and sigma_i and sens_i are rounded to 3 significant digits before use (the table prints them rounded). normal() is the book's Box-Muller with the held partner: tensor 0's normal() draws two uniforms and holds the sine partner, tensor 1's normal() returns the held value without drawing, tensor 2's draws two uniforms again, and so on (the held value carries across tensors). The estimated loss increase of tensor i stored at n bits is E_in = (1/2) sens_i N_i sigma_i^2 rel_n, its bytes N_i (n + 0.5) / 8.

| i | N_i | sigma_i | sensitivity sens_i |
|---|---:|---:|---:|
| 0 | 12,288 | 0.0402 | 7.35 |
| 1 | 12,288 | 0.0278 | 8.65 |
| 2 | 12,288 | 0.0262 | 39.9 |
| 3 | 4,096 | 0.0171 | 22.9 |
| 4 | 8,192 | 0.0209 | 88 |
| 5 | 4,096 | 0.018 | 13.7 |
| 6 | 8,192 | 0.022 | 1.25 |
| 7 | 12,288 | 0.026 | 51.7 |
| 8 | 4,096 | 0.0264 | 3.7 |
| 9 | 16,384 | 0.0181 | 3.1 |
| 10 | 8,192 | 0.0124 | 11.8 |
| 11 | 8,192 | 0.0149 | 72.6 |
<!-- /worked -->

The algorithm is greedy. Every tensor starts at its cheapest width. An **upgrade** moves a tensor to a wider option, and its value is the error it removes per extra byte. As long as some upgrade fits in the bytes left, take the one with the best value:

```text
record Option:
    bytes: real
    error: real
    bits: int

function greedy_plan(options: list<list<Option>>, budget: real) -> list<int>:
    # options[i]: tensor i's options, cheapest first, each strictly more accurate than the one before
    T = len(options)
    at = zeros(int, [T])
    used = sum([options[i][0].bytes for i in 0 .. T])
    require used <= budget, "the budget is below the cheapest plan"
    loop:
        best_gain = 0.0
        best_i = -1
        best_to = -1
        for i in 0 .. T:
            for to in at[i] + 1 .. len(options[i]):
                extra = options[i][to].bytes - options[i][at[i]].bytes
                if used + extra <= budget:
                    gain = (options[i][at[i]].error - options[i][to].error) / extra
                    if gain > best_gain:
                        best_gain = gain
                        best_i = i
                        best_to = to
        if best_i < 0:
            break
        used = used + options[best_i][best_to].bytes - options[best_i][at[best_i]].bytes
        at[best_i] = best_to
    return [options[i][at[i]].bits for i in 0 .. T]
```

The upgrade to any wider option is considered, not only the next, because a jump of two steps can pay when the next step alone does not. Greedy is not exact in general. Whether it is close, we test on a tiny instance where every assignment can be tried. Six tensors with four widths each have $4^6 = 4096$ assignments:

<!-- worked: quantization.greedy_vs_exhaustive -->
The first six tensors of the previous table (sizes 12,288, 12,288, 12,288, 4,096, 8,192, 4,096), tiers n in {2, 3, 4, 8} bits (4096 assignments). Budget = bits per weight (mean over the six, scales included) x total weights / 8 bytes. 'plan' lists the bits of tensors 0 .. 5. Greedy: start every tensor at 2 bits, then repeatedly take the upgrade that removes the most estimated error per extra byte among those that still fit. Exhaustive: the plan of least total estimated error among all that fit. 'ratio' = greedy error / exhaustive error.

| budget bits/weight | budget bytes | greedy plan | greedy bytes | greedy error | exhaustive plan | exhaustive bytes | exhaustive error | ratio |
|---:|---:|---|---:|---:|---|---:|---:|---:|
| 2.75 | 18,304 | 222332 | 18,176 | 134.5532 | 222332 | 18,176 | 134.5532 | 1.0000 |
| 3.25 | 21,632 | 323332 | 21,248 | 42.6314 | 323332 | 21,248 | 42.6314 | 1.0000 |
| 3.75 | 24,960 | 333443 | 24,832 | 16.4155 | 333443 | 24,832 | 16.4155 | 1.0000 |
| 4.25 | 28,288 | 434443 | 27,904 | 6.4040 | 434443 | 27,904 | 6.4040 | 1.0000 |
| 4.75 | 31,616 | 444444 | 29,952 | 4.3223 | 444444 | 29,952 | 4.3223 | 1.0000 |
| 5.25 | 34,944 | 444484 | 34,048 | 2.8556 | 444484 | 34,048 | 2.8556 | 1.0000 |
| 5.75 | 38,272 | 444888 | 38,144 | 2.6432 | 448844 | 38,144 | 2.6270 | 1.0062 |
| 6.50 | 43,264 | 448884 | 42,240 | 1.1603 | 448884 | 42,240 | 1.1603 | 1.0000 |
| 7.50 | 49,920 | 848884 | 48,384 | 0.4805 | 848884 | 48,384 | 0.4805 | 1.0000 |

The nine budgets are a sample. The same instance on a finer sweep, every budget from 2.50 to 8.50 bits per weight in steps of 0.05 (121 budgets): greedy equals the exhaustive optimum at 96 of them and is worse at 25, by at most a factor 1.7543 (budgets 7.90 to 8.15); at 6.00 bits per weight the factor is 1.5872.
<!-- /worked -->

In 8 of the 9 budgets the greedy plan is the exhaustive optimum. At 5.75 bits per weight it is not: it ends with 8 bits on tensors 3, 4 and 5 (`444888`) where the optimum puts them on tensors 2 and 3 only (`448844`), and its error is 0.62% higher. The cause is the general one of the knapsack: the value ratio ranks the upgrades by efficiency and not by what fills the budget best, so the last step can leave bytes
that a different set of steps would have used. For items whose benefit falls steadily with cost (convex options) and a budget that lands exactly on a step, greedy is optimal. Elsewhere it can be far from the optimum: the table above is a sample of nine budgets, and the finer sweep under it finds greedy off by up to a factor 1.75. The "within one upgrade" that is sometimes quoted for greedy is a statement about the linear relaxation of the problem, in which the last upgrade may be taken in part; it says nothing about the best whole assignment. On the twelve tensors:

<!-- worked: quantization.greedy_plan -->
The twelve tensors of the input table, tiers n in {2, 3, 4, 6, 8}, budget 4.5 bits per weight x 110,592 weights / 8 = 62,208 bytes. A random plan draws n for tensors 0 .. 11 in order as TIERS[integer(5)] with TIERS = (2, 3, 4, 6, 8) from the generator, seed 14439535479056496435 (seed_of 'quantization.random_plans', defined on the pseudocode page), and is kept when it fits (drawing until 200 are kept; 976 were drawn).

| i | N_i | sens_i | sigma_i^2 sens_i | chosen bits | bytes | estimated error |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 12,288 | 7.35 | 1.188e-02 | 4 | 6,912 | 0.6819 |
| 1 | 12,288 | 8.65 | 6.685e-03 | 4 | 6,912 | 0.3838 |
| 2 | 12,288 | 39.9 | 2.739e-02 | 4 | 6,912 | 1.5724 |
| 3 | 4,096 | 22.9 | 6.696e-03 | 6 | 3,328 | 0.0066 |
| 4 | 8,192 | 88 | 3.844e-02 | 6 | 6,656 | 0.0752 |
| 5 | 4,096 | 13.7 | 4.439e-03 | 4 | 2,304 | 0.0849 |
| 6 | 8,192 | 1.25 | 6.050e-04 | 3 | 3,584 | 0.1260 |
| 7 | 12,288 | 51.7 | 3.495e-02 | 4 | 6,912 | 2.0064 |
| 8 | 4,096 | 3.7 | 2.579e-03 | 4 | 2,304 | 0.0493 |
| 9 | 16,384 | 3.1 | 1.016e-03 | 3 | 7,168 | 0.4230 |
| 10 | 8,192 | 11.8 | 1.814e-03 | 4 | 4,608 | 0.0694 |
| 11 | 8,192 | 72.6 | 1.612e-02 | 4 | 4,608 | 0.6169 |

| plan | bytes | bits per weight | estimated error | error / greedy |
|---|---:|---:|---:|---:|
| greedy plan | 62,208 | 4.500 | 6.0959 | 1.00 |
| best uniform tier that fits (4 bits) | 62,208 | 4.500 | 7.1653 | 1.18 |
| median of 200 random plans that fit |  |  | 110.8890 | 18.19 |
| all tensors at 8 bits | 117,504 | 8.500 | 0.0218 |  |
<!-- /worked -->

The value of an upgrade is independent of the size of the tensor (error and bytes both scale with $N_i$) and proportional to $\mathrm{sens}_i\sigma_i^2$, so the two tensors with the least $\mathrm{sens}_i\sigma_i^2$ per weight (6 and 9) end at 3 bits, the others at 4, and the two small tensors 3 and 4 get 6 bits with the bytes that were left over when no upgrade of a large tensor fitted. The best uniform width that fits the same bytes (all tensors at 4 bits) has 1.18 times greedy's error, 7.1653 against 6.0959. The median random plan that fits the budget has 18.2 times greedy's error, because chance puts 2-bit codes on tensors that cannot take them: where the bits are placed matters more than how many there are.
These errors are estimates from a model. A real search must measure the plan on held-out data before accepting it, and the next section shows that brain does exactly that.

## What a careful quantizer counts, chooses and checks

The sections above leave a short list of habits. A format is compared by the bits it stores, scales included: a 4-bit code with a 16-bit scale per 32 weights is 4.5 bits per weight, not 4, and formats are compared at equal $\mathrm{bpw}$. The group is a divisor of the reduction axis and never straddles a row, because a group that crosses a row boundary shares one scale between two output channels; a tensor whose reduction length is not a whole number of groups is kept in floating point, and a quantizer refuses it rather than guess. Codes are chosen against the stored scales, since the decoder only sees what is stored.

Weights use signed codes with a symmetric range, which keeps the product free of correction terms, and an offset only where a group is not centred; a floor per sub-block removes the cost of an off-centre group at one 6-bit number. Packing follows a single stated order and is tested on a negative code, because a nibble read as unsigned instead of signed changes only the negative weights, and a decoder that does so passes any test that has only positive values. Error is reported as MSE or SQNR together with cosine, and the test is on the whole output, since the error that counts is the one a layer's output sees and ultimately the held-out loss of the whole model. A speed is reported as counted bytes and FLOPs, and any time is measured the way Chapter 11 says.

## Quantization in brain: decoders, encoders, tiers, import and search

**The file format layer.** `crates/checkpoint/src/gguf.rs` reads the block formats that GGML originated and that GGUF files carry. `GgmlType` lists each format's weights per block (`block_elems`) and bytes per block (`block_bytes`): Q4_0 is 18 bytes for 32 weights, Q8_0 34, Q4_K 144 for 256, Q5_K 176, Q6_K 210, the numbers of the byte table above. The decoders are one function per format
(`deq_q4_0`, `deq_q8_0`, `deq_q4_k`, `deq_q5_k`, `deq_q6_k`) and they are the specification. `deq_q4_k` is this chapter's `dequantize_q4k` line for line: a 12-byte scale field unpacked by `scale_min_k4`, then for each of four chunks of 32 bytes the low nibbles with the scale of sub-block $2c$ and the high nibbles with that of $2c+1$; Q4_0's two halves of 16 are in `deq_q4_0`.
The decoders' output is `d * sc * q - dmin * m` for the affine formats and `d * q` for the symmetric ones. `deq_q6_k` reads its sixteen scales as signed bytes and subtracts 32 from each 6-bit code.
`tensor_range` in the same file decodes only the blocks that a range of elements touches, which is possible because each block is independent: block $b$ starts at byte $b \times$ `block_bytes`.

**The encoder.** `crates/checkpoint/src/quant.rs` is the inverse of every decoder: its module comment says that each `deq_*` function is the specification and the quantizer's job is to choose the integers that land the formula closest to the input. For the affine formats `fit_affine_unsigned` alternates between
assigning codes and re-solving the step and floor by least squares, from a bank of starting steps, and `quantize_positive` is this chapter's second level (`max(values) / 63`, each value divided and rounded). The K-quant fit (`fit_k4_or_k5`) re-assigns every sub-block's codes "against the FINAL quantized effective scale/floor", which is the rule of the section on one super-block. `round_trip_stats` returns the RMSE and the cosine of a quantize-decode pass, the two numbers this chapter asks for. The Q4_0 encoder maps the signed element of largest magnitude to code 0 by `d = extreme / -8`
(`biased_extreme_scale`): exactly the 16-level arrangement of the table, and a different arrangement from the 15-level one of the 4-bit tier below.

**brain's own tiers.** `crates/model/src/int8.rs` quantizes a weight to int8 in groups of `GROUP = 32` along the reduction axis, with scale `max|block| / 127` stored as f32, so that its bytes per weight are $1 + 4/32 = 1.125$ (`Weight::bytes_for` in `crates/model/src/ops.rs`, the 9 bits per weight that `Budget::BitsPerWeight` of `modelopt` names).
`crates/model/src/int4.rs` does the same at four bits with `max|block| / 7` and eight nibbles to a 32-bit word, so $0.5 + 0.125 = 0.625$ bytes per weight, the figure Chapter 3 used for its byte count. Its decode takes a nibble, shifts it into the top of a byte and shifts it back arithmetically to sign-extend; the test
`dequantize_weight_q4_sign_handling_is_load_bearing` decodes the same codes with the sign dropped and requires the result to differ, because, as its comment says, 4-bit sign extension is the easiest thing to get wrong.

**GGUF import.** `crates/gguf/src/kquant.rs` relays a tensor in any of six block formats (Q4_K, Q5_K, Q6_K, Q5_0, Q4_0, Q8_0) into one device layout without ever decoding to floating point: packed codes, a per-group pair of 8-bit sub-scales (`wsm`) and a per-super-block pair of f16 factors (`wd`),
whose product is rebuilt on the device as `ds * code - dm`. Its test (`crates/gguf/tests/it/kquant.rs`) checks that the reconstruction equals the file decoder's output with `assert_eq!` on the floats, not with a tolerance. `crates/gguf/src/int8_direct.rs` is the same idea for Q8_0, which becomes brain's int8 layout by repacking bytes,
because `GROUP = 32` equals Q8_0's own block and both scales are `max|x| / 127`.
The affine kernels use the factored product of this chapter: the shader `crates/kernels/wgsl/quant_group_sum.wgsl` computes the activation-only group sum once (there on the int8-packed activation, so that the sum is exact; its header calls it "independent of n") so that no output column recomputes it.

<!-- worked: quantization.device_layouts -->
Device bits per weight = 8 (slot / 8 + 2 / group + 4 / spb): the codes in a slot of `slot` bits, two bytes (sc, m) per group, four bytes (the f16 pair d, dmin) per `spb` weights. The file's bits per weight are 8 x block bytes / block weights.

| format | slot bits | group | spb | device bits per weight | file bits per weight | device / file |
|---|---:|---:|---:|---:|---:|---:|
| Q4_K | 4 | 32 | 256 | 4.6250 | 4.5000 | 1.028 |
| Q5_K | 8 | 32 | 256 | 8.6250 | 5.5000 | 1.568 |
| Q6_K | 8 | 16 | 256 | 9.1250 | 6.5625 | 1.390 |
| Q4_0 | 4 | 32 | 32 | 5.5000 | 4.5000 | 1.222 |
| Q5_0 | 8 | 32 | 32 | 9.5000 | 5.5000 | 1.727 |
| Q8_0 | 8 | 32 | 32 | 9.5000 | 8.5000 | 1.118 |

The layout this replaced kept one f32 pair (scale, min) per group, 8 bytes per group of 32: for Q4_K that is 6.0 bits per weight (4 for the codes, 2 for the pair), against 4.625 now.
<!-- /worked -->

**The search.** `crates/modelopt` is this chapter's last section as code. `sensitivity` computes, for each linear (a site, or the tensors an engine fuses, which share a tier) and each candidate tier, $\tfrac12\sum_j F_j (w_j - Q(w_j))^2$ where $Q$ is the deployed codec (`Weight::round_trip`) and
$F$ is the mean over calibration batches of the squared gradient, the diagonal empirical Fisher standing for the curvature of the loss: $\mathrm{sens}_i$ made concrete, per weight. `assign` in `solve.rs` is `greedy_plan`: it drops options that cost more and are no more accurate, starts every site at its cheapest option, ranks upgrades by error removed per share of the limit,
and its comment states the guarantee for the linear relaxation (exact on convex options when the budget lands on a breakpoint, within one upgrade of the relaxation otherwise), which is weaker than optimality of the whole assignment, as the sweep above shows. It also supports a second limit on measured product time, by weighting the limits. `search_weights` then does what the estimate cannot: it applies the plan by simulated quantization, scores it on held-out batches that are
required to be disjoint from the calibration batches (a test checks that overlapping sets are refused), and compares it with every uniform tier that fits the budget. The acceptance test builds a small trained model, searches at a budget halfway between uniform int8 and uniform 4-bit, and asserts that the plan meets
the budget, has a lower held-out loss than every uniform tier that fits and than the median of shuffles of its own tiers among sites of the same shape. The candidate tiers are those `Weight::packs` accepts (int8, 4-bit, NF4, fp8 and the 16- and 32-bit floats), so the K-quant tiers of the import path are not among them.

### One outlier and the scale of a whole row

Until August 2026 brain's int8 and int4 quantizers emitted one scale per output row, with the reduction axis of the models it served reaching 17,408 weights. Commit f9c3977e7 (28 August 2026) replaced this by a scale per group of 32 for every consumer, deleting the old path rather than keeping it. Its message
names the defect as the one of this chapter's outlier section and the reason for 32: it is Q8_0's block size. The measurement behind it is the repository's test `group_wise_confines_a_row_outlier_that_whole_channel_spreads_over_the_row` (in `crates/model/src/int8.rs`): two rows of 512 weights, one weight replaced by 100.0. Re-implemented from its description:

<!-- worked: quantization.repo_outlier_fixture -->
Two rows of k = 512 weights, w_i = (i mod 37) 0.017 - 0.3 for i = 0 .. 1023 (row-major), with the weight at column 509 of each row replaced by 100.0. 8-bit symmetric codes, scale = max|group| / 127 (floored at 1e-8), q = clamp(round(w / scale), -127, 127) with ties away from zero, w' = scale q, error |w' - w|. 'largest error outside' is over every weight not in the outlier's own group of 32; 'destroyed' counts weights other than the outlier with error above 0.1.

| scales | largest error outside the outlier's group | weights destroyed | of |
|---|---:|---:|---:|
| one scale per 32 | 0.001189 | 39 | 1022 |
| one scale per row | 0.312000 | 687 | 1022 |

Ratio of the two largest errors: 262.
<!-- /worked -->

The numbers equal those of knowledge #65 to the digit: with one scale per row, the largest error outside the outlier's group is 0.312 and 687 of 1022 weights are destroyed; with a scale per 32 weights it is 0.001189 and 39 (only the outlier's own group). The lesson of that entry is the general one, that a bound should be set by the block and not by the length of the row, so a test asserts boundedness and not a ratio.
Two consequences are visible in the code. Because the group is now 32, a scale tensor of the old row-wise shape would still parse as the first row's group scales and produce plausible garbage, so `check_scale_len` recognises exactly that length and says the format changed. And the storage ceiling of int8 moved from 4 times smaller than f32 to $4/1.125 = 3.56$: the scale is not free.

### Keeping the two-level structure on the device

Commit 76161efc9 (2 September 2026) added the relayout of GGUF blocks into a device layout, with the first version storing the product of the two levels, one f32 `(scale, min)` pair per group of 32. Commit c0954e8df (3 September 2026) replaced it with the two pieces the file already carries: a pair of bytes per group
and an f16 pair per super-block. The table above gives the effect on the bytes of Q4_K, computed from the layout: 6.0 bits per weight with the flat pair against 4.625 with the two-level one, and its last column is the reason the same commit left Q5_K's footprint at 1.57 times the file's: the layout stores a 5-bit code in an 8-bit slot, and the follow-up
design to fix it was recorded and not built. The lesson is that a quantized format's structure is information: storing the product of the two levels throws it away and costs bytes in the very place where bytes are what is being bought.

### The estimate chooses, the measurement accepts

Commit e68da7686 (4 November 2026) added `modelopt` with one rule above the rest: sensitivities are additive estimates that pick candidates, and only a whole-model measurement on held-out data accepts a result. The rule is in the code (`search_weights` returns the plan together with the held-out loss of the original and of each uniform baseline,
and `beats_baselines` is the comparison a plan has to win) and is stated in the invariants of `.agents/roadmap/modelopt.md`. The ledger there also records, among its runs, one where the plan was refused because the measurement of a code model on prose text disagreed with the second-order estimate (which assumes the mean gradient is near zero);
The book has not re-run it and cites it only for the process: a search that can say no. Commit b698ce353 (4 August 2026), which wrote the encoders for every format, records the same discipline at a smaller scale: a Q3_K sub-block multiplier quantized with the wrong bit width had silently clamped nearly every scale to the edges of its field until a test caught it.
The current code carries the fix (`quantize_signed_biased` takes the width of the on-disk field).

## Check your understanding

Answers are at the end of the chapter.

1. **Find the error.** "A 4-bit code with step $\Delta$ makes an error of at most $\Delta/2$, so its mean squared error is $\Delta^2/4$." What is the mean squared error under the uniform-error model, and why is the claim wrong?
2. **Predict.** A tensor of 4,096 weights has magnitude at most $0.03$ except for one weight, $0.6$. It is stored in 4-bit symmetric codes with one scale for the tensor. What is the step, and what does every small weight decode to?
3. **Find the error.** "Splitting a tensor into ever smaller groups always raises the accuracy for the stored bits." Use the tables of this chapter to say where it fails.
4. **Predict.** A Q4_K-like format uses sub-blocks of 16 weights instead of 32, with the same field widths (6-bit scale and 6-bit min per sub-block, f16 $d$ and $d_{\min}$ per 256 weights, 4-bit codes). How many bits per weight does it store?
5. **Explain.** Why is a weight-only quantized product faster in a decode step at batch size 1 and not in a large batch, with the same bytes saved in both?
6. **Predict.** The weights of a layer have an SQNR of 20 dB after quantization. The layer has 4,096 inputs instead of 256. What do you expect of the SQNR of its output, and what assumption does the answer rest on?
7. **Find the error.** "Greedy search by error removed per byte is optimal, because it always makes the best move." Use the six-tensor table to refute it.
8. **Explain.** After the 6-bit scales and mins of a Q4_K block have been rounded, why must the codes be chosen again?

## Summary

A weight is stored as a small integer code and read back through a scale and, when the group is not centred on zero, a zero point: affine quantization, $w' = \Delta(q - q_0)$. The error of a code is bounded by half a step and, when the density of the weights is flat over a step, has the mean square $\Delta^2/12$; each added bit quarters the error power and adds 6.02 dB (measured: 7.3 dB from 2 to 3 bits, 6.09 dB from 7 to 8), and the model fails only when the step is as large as the spread of the data (2 bits). A scale taken from the largest weight of a tensor is set by its most extreme member: 58 outliers among 16,384 weights
left 81.7 percent of the ordinary weights at zero. Group-wise scales confine an outlier to its group, at a price of scale bits per weight, $n_b + (\text{scale bits})/G$, and the break-even between smaller groups and wider codes is visible in the tables. Weight-only quantization keeps activations in floating point, reads a
fraction of the bytes (4.5 bits against 16 is 3.56 times fewer, 14.000 GB to 3.938 GB for 7 billion weights), and factors the scale out of the dot product so that decoding adds about $3/G$ FLOPs per weight. K-quantization quantizes the scales again: Q4_K stores 256 weights in 144 bytes (4.5 bits per weight) as eight 32-weight sub-blocks with 6-bit scales and mins and two f16 factors,
in a packing whose rules (the twelve scale bytes, the low and high nibbles of a chunk) are exact enough to decode by hand; the codes must be chosen against the stored scales. Integer and floating-point codes differ in their error profile, absolute for the first and relative for the second, and the output error of a layer follows the weights' SQNR when the errors are independent of the inputs. Last, tensors differ in sensitivity,
and a greedy knapsack on error removed per byte places the bits: optimal in 8 of 9 sampled budgets on a tiny instance and 0.62 percent off in the ninth (a finer sweep of 121 budgets finds it off at 25 of them, by up to 1.75 times), and on a larger one the best uniform width has 1.18 times its error. brain applies the same ideas in the decoders and encoders of `checkpoint`, in the relayout of `gguf`, in its group-wise int8 and int4 tiers and in the measured search of `modelopt`, which accepts nothing that the held-out loss does not confirm.

The weights are now small; the activations and the arithmetic are still floating point. The next chapter, Neural Processing Units and Graph Compilers, quantizes the activations as well, calibrates the ranges on data (post-training quantization), and compiles the whole network into integer arithmetic for a fixed-function accelerator.

## References

- Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew Howard, Hartwig Adam and Dmitry Kalenichenko, "Quantization and training of neural networks for efficient integer-arithmetic-only inference", 2018.
- Elias Frantar, Saleh Ashkboos, Torsten Hoefler and Dan Alistarh, "GPTQ: Accurate post-training quantization for generative pre-trained transformers", 2023.
- Tim Dettmers, Mike Lewis, Younes Belkada and Luke Zettlemoyer, "LLM.int8(): 8-bit matrix multiplication for transformers at scale", 2022.
- Bernard Widrow and Istvan Kollar, "Quantization noise: roundoff error in digital computation, signal processing, control, and communications", 2008.

## Exercises

1. **[Calculate]** For the weights $0.31, -0.12, 0.05, 0.22, -0.40, 0.18, 0.00, 0.27$ find the symmetric 4-bit scale and codes ($q_{\max} = 7$) and the affine scale, zero point and codes (16 codes), with ties to even. Which decodes with the smaller largest error, and is it within $\Delta/2$?
2. **[Calculate]** Derive the SQNR of an $n$-bit code of a signal that is uniform on $[-A, A]$ (the code covering exactly $[-A, A]$ with $2^n$ levels) and show that it is $6.02\,n$ decibels exactly. Explain why a Gaussian signal gives a smaller constant.
3. **[Calculate]** A 70-billion-weight model is stored in Q8_0, Q6_K, Q5_K and Q4_K. Give the gigabytes of each and the bytes that one token reads at batch size 1. At a bandwidth of 1,000 GB per second, give the roofline bound of Chapter 3 on tokens per second for each.
4. **[Calculate]** Count the bytes of a Q5_K super-block from its field list (the layout adds a 32-byte plane of fifth bits) and its bits per weight, and give the position in `qh` of the fifth bit of weight 37.
5. **[Calculate]** The twelve scale bytes of a block are `02 07 0C 11 16 1B 20 25 2A 2F 34 39`, in hexadecimal. Unpack the eight pairs $(\mathrm{sc}_j, \mathrm{m}_j)$ with `scale_min`.
6. **[Calculate]** Three tensors have options (bytes, error): A: (10, 9), (20, 4), (40, 1); B: (10, 6), (20, 3), (40, 2.5); C: (10, 3), (20, 2.5), (40, 0.5). The budget is 70 bytes. Run `greedy_plan` by hand, and then find the exhaustive optimum. Do they agree?
7. **[Implement]** Implement `affine_params`, `quantize_affine` and `dequantize_affine`, the SQNR, and the book's SplitMix64 generator, and reproduce `quantization.eight_weights`, `quantization.uniform_error_model`, `quantization.outliers` and `quantization.group_size_tradeoff`.
8. **[Implement]** Implement `quantize_q4k`, `scale_min`, the packing of the twelve scale bytes (the inverse of `scale_min`) and `dequantize_q4k`, and reproduce `quantization.q4k_inputs`, `quantization.q4k_scales`, `quantization.q4k_bytes`, `quantization.q4k_walk`, `quantization.q4k_check` and `quantization.q4k_reassign`. Then change the rule to find the codes against the ideal steps and confirm the last table.
9. **[Implement]** Implement `greedy_plan` and the exhaustive search and reproduce `quantization.rel_table`, `quantization.search_inputs`, `quantization.greedy_vs_exhaustive` and `quantization.greedy_plan`. Find a budget on the tiny instance at which greedy is worse than in the table, if one exists, by sweeping the budget in steps of 0.05 bits per weight.
10. **[Implement]** Implement the E4M3 and int8 group codes and the layer experiment, and reproduce `quantization.fp8_vs_int`, `quantization.layer_output` and `quantization.factored_dot`.
11. **[Investigate]** Using the generator and weights of `quantization.outliers`, vary the outlier probability from 0.0005 to 0.02 and the factor from 5 to 50. For each setting find the group size at which the 4-bit SQNR of the ordinary weights is highest per stored bit. Is there a setting where one scale per 8 weights loses to one per 64?
12. **[Investigate]** From the rows of `quantization.device_layouts`, say which formats pay most, relative to the file, for the layout they are held in, and why. What would a layout with a 4-bit plane and a separate 1-bit plane for Q5_K cost per weight (the scales and factors as in the table), and how would that compare with the 1.57 of the table?
13. **[Investigate]** Read `crates/modelopt/src/sensitivity.rs` and `solve.rs`. State, for each of the three hypotheses of the estimate (second order, diagonal curvature, sites independent), a situation in which it fails, and say what in `search_weights` guards against each failure.

## Key terms

- **affine quantization**: the map from a real number to an $n_b$-bit integer code and back, $q = \mathrm{clamp}(\mathrm{round}(w/\Delta) + q_0, 0, 2^{n_b} - 1)$ and $w' = \Delta(q - q_0)$, a multiplication and an offset applied to a group of weights. <!-- concept: affine-quantization -->
- **scale and zero point**: the pair $(\Delta, q_0)$ of a group: the distance between neighbouring levels and the code that stands for the number 0, chosen so that the codes cover the group's range and 0 is exactly a code. <!-- concept: scale-and-zero-point -->
- **group-wise quantization**: giving each run of $G$ consecutive weights its own scale, so that one large weight sets the step for its group only; it costs $(\text{scale bits})/G$ bits per weight in addition to the codes. <!-- concept: group-wise-quantization -->
- **weight-only quantization**: storing the weights as low-bit codes with scales while the activations stay in a higher-precision format, decoding the codes inside the matrix product so that the bytes read, and not the arithmetic, shrink. <!-- concept: weight-only-quantization -->
- **k-quantization**: a family of block formats in which a super-block of weights is divided into sub-blocks whose scales and mins are themselves stored as small integers times one or two shared factors (two levels of scale), as in Q4_K: 256 weights, eight sub-blocks, 144 bytes. <!-- concept: k-quantization -->
- **quantization error**: the difference $e = w' - w$ between a decoded weight and the original, bounded by half a step, and summarized by its mean squared error or the signal-to-quantization-noise ratio $10\log_{10}(\mathrm{mean}(w^2)/\mathrm{MSE})$ in decibels. <!-- concept: quantization-error -->
- **mixed-precision weight search**: choosing the storage width of each tensor so as to minimize the estimated total loss increase within a byte budget, using each tensor's sensitivity; a knapsack problem solved greedily and accepted only after measuring the whole model. <!-- concept: mixed-precision-weight-search -->

## Answers

**Recall questions.**

1. $-37 \times 0.01 = -0.37$; the largest error is half a step, $0.005$.
2. $448$. The format spans about five decades ($448/2^{-9} \approx 2.3 \times 10^5$), less than a tensor's weights, so a scale moves each block's largest value to the top of the format.
3. Half the spacing, $h/2$; for a floating-point format that is the rounding error, whose relative size is at most the unit round-off $u$.
4. $(0.01 + 0.04 + 0.09)/3 = 0.0467$; the square root, $0.216$, is in the units of the quantity predicted.
5. $7 \times 10^9 \times 2 = 14 \times 10^9$ bytes, 14 GB. The bandwidth of memory, because the intensity (1 FLOP per byte) is below the ridge point.

**Check your understanding**

1. Under the uniform-error model $\mathrm{MSE} = \Delta^2/12$. $\Delta/2$ is the largest error, reached only for a weight exactly half a step from a level; the average of $e^2$ over $[-\Delta/2, \Delta/2]$ is a third of the largest value $(\Delta/2)^2$, that is $\Delta^2/12$. The claim squares the worst case, which is 3 times too large.
2. $\Delta = 0.6/7 = 0.0857$. A small weight has magnitude at most $0.03 < \Delta/2 = 0.0429$, so every small weight decodes to $0$ (only the large one survives).
3. It fails when the scales cost more than they buy. Table of the group sizes: 4-bit codes in groups of 4 cost 8.0 bits per weight and reach 24.54 dB on the heavy-tailed weights, while 8-bit codes with one scale for the tensor cost 8.002 and reach 28.56 dB.
4. $256 \times 4 = 1024$ bits of codes, 16 sub-blocks $\times$ 12 bits $= 192$, plus $32$ for the two f16 factors: $1248/256 = 4.875$ bits per weight.
5. At batch size 1 the intensity is about 2 FLOPs per weight over the bytes of a weight, below the ridge point, so the time follows the bytes and 3.56 times fewer bytes is up to that factor in the bound. In a large batch each weight read serves many rows, the intensity is above the ridge, the product is limited by the arithmetic, which is unchanged, and the saved bytes buy nothing.
6. About 20 dB: the output SQNR equals the weights' when the errors are independent of each other and of the inputs, because the number of inputs cancels from the ratio (the table showed 20.32 against 20.39 for 256 inputs). The assumption is independence; real activations with large channels can break it.
7. At 5.75 bits per weight greedy gives `444888` with error 2.6432 and the exhaustive optimum is `448844` with 2.6270. Each greedy move was the most efficient one available, but the sequence of such moves does not have to fill the budget best.
8. The decoder reads the stored $d\,\mathrm{sc}_j$ and $d_{\min}\mathrm{m}_j$, not the ideal steps and floors; codes found against the ideal ones are decoded with different levels and the error is higher (MSE $9.5759 \times 10^{-6}$ against $9.0587 \times 10^{-6}$ for the block of the chapter).
