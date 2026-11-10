<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Promptable Segmentation and Video Tracking

*How can one model segment any object you point at, and follow it through a video?*

Someone looks at the picture below and says "this one", pointing at the pixel in row 2, column 2. What did they mean? The segmenter of Chapter 5 would answer with a class for every pixel, and it could not even be asked, because its classes were fixed when it was trained. Here the question arrives with the picture, and the answer has to be a set of pixels. To see how little a click settles, build the simplest thing that takes one: grow a region outward from the clicked pixel, adding a 4-neighbour whenever its code is within a tolerance of the clicked code.

```text
function grow(image: array<int>[n, n], r0: int, c0: int, tol: int) -> array<int>[n, n]:
    require 0 <= r0 and r0 < n and 0 <= c0 and c0 < n
    ref = image[r0, c0]
    inside = zeros(int, [n, n])
    inside[r0, c0] = 1
    changed = true
    while changed:
        changed = false
        for r in 0 .. n:
            for c in 0 .. n:
                if inside[r, c] == 0 and abs(image[r, c] - ref) <= tol:
                    next_to = (r > 0 and inside[r - 1, c] == 1) or (r < n - 1 and inside[r + 1, c] == 1) or (c > 0 and inside[r, c - 1] == 1) or (c < n - 1 and inside[r, c + 1] == 1)
                    if next_to:
                        inside[r, c] = 1
                        changed = true
    return inside
```

<!-- worked: segment_anything.toy_prompts -->
The image is 8 by 8 with the codes shown (0 is background). The toy segmenter takes a click (row 2, column 2, the code 9) and a tolerance, and grows the set of pixels reachable through 4-neighbours whose code differs from the clicked code by at most the tolerance. The last column adds a second, negative click at (row 3, column 6) on the right-hand object, grown with tolerance 0 and removed from the tolerance-6 mask.

| row | image | tolerance 0 | tolerance 3 | tolerance 6 | tolerance 6 minus the negative click |
|---|---|---|---|---|---|
| 0 | 00000000 | ........ | ........ | ........ | ........ |
| 1 | 06666000 | ........ | .####... | .####... | .####... |
| 2 | 06996330 | ..##.... | .####... | .######. | .####... |
| 3 | 06996330 | ..##.... | .####... | .######. | .####... |
| 4 | 06666330 | ........ | .####... | .######. | .####... |
| 5 | 00000000 | ........ | ........ | ........ | ........ |
| 6 | 00000000 | ........ | ........ | ........ | ........ |
| 7 | 00000000 | ........ | ........ | ........ | ........ |
| count |  | 4 | 16 | 22 | 16 |
<!-- /worked -->

The same click gives three masks as the tolerance rises: the bright core of 4 pixels, the whole left object of 16, and the left object with a piece of its neighbour glued on, 22. A second click, this time on the neighbour and marked as *not* the object, removes the glue and returns the 16 pixels. The segmenter is the same program throughout and the masks differ because the **prompt** differs. That is what **promptable segmentation** means: a model whose input is an image together with a hint about which object is wanted, a point, a box or a rough mask, and whose output is the mask of that object. One trained model, any object, no class list.

Nothing in the toy is learned, and a learned model has a harder job, because the tolerance in the toy is chosen by the person who calls it. A learned model has to decide for itself how far "this one" extends. The table below puts a number on how much is at stake. Two people give the same click and mean different things, one the core and one the whole object, and each candidate mask is scored against each reading by the intersection over union of Chapter 4, taken over sets of pixels as in Chapter 5.

<!-- worked: segment_anything.toy_readings -->
Overlap (intersection over union of pixel sets, as in Chapter 5) of each candidate with the two readings of the click.

| candidate | pixels | IoU with the core (4 pixels) | IoU with the whole object (16 pixels) |
|---|---:|---:|---:|
| tolerance 0 | 4 | 1.000 | 0.250 |
| tolerance 3 | 16 | 0.250 | 1.000 |
| tolerance 6 | 22 | 0.182 | 0.727 |
| tolerance 6, negative click | 16 | 0.250 | 1.000 |

If the user means either reading with probability 1/2, the per-pixel average of the two truth masks is 1 on the 4 core pixels, 1/2 on the 12 other pixels of the object and 0 elsewhere: the squared-error optimum of one mask is a blurred mask that neither reading would draw.
<!-- /worked -->

Each candidate is perfect for exactly one reading and poor for the other: 0.25 for the wrong one, because 4 pixels are inside 16. A single mask trained on both kinds of answer would do worse than either, because the squared-error optimum is the average of the two truth masks, a mask that is 1 on the core, one half on the other 12 pixels of the object and so draws neither. The way out that the model of this chapter takes has three parts, each of which gets its own section. It answers a single click with several masks, so that each reading can have its own. It predicts, for every mask, the overlap that mask is expected to have with the truth, so that a program (or a person) can choose among them. And it is trained with a loss that charges only the best of the masks, so that the three heads are free to specialize instead of each trying to be the average.

Five ideas from earlier chapters come first; the answers are at the end.

1. (Volume V, [Chapter 6](../v05-sequences/06-seq2seq.md), cross-attention.) In a cross-attention layer with 7 queries and 16 keys, where do the queries, keys and values come from, and what is the shape of the weight matrix?
2. (Volume VII, [Chapter 6](06-vision-transformers.md), windowed attention.) A 64 by 64 grid of tokens is cut into windows of 14 by 14 after zero-padding to a multiple of 14. How many windows are there and how many token rows does the padded grid have?
3. (Volume VII, [Chapter 4](04-detection.md), intersection over union.) A mask of 4 pixels lies inside a mask of 16. What is their IoU?
4. (Volume V, [Chapter 8](../v05-sequences/08-positions.md), rotary and sinusoidal position.) A query and a key are rotated by angles proportional to their positions $i$ and $j$. What does their dot product depend on?
5. (Volume VII, [Chapter 2](02-convolution.md), transposed convolution.) A transposed convolution with a $2 \times 2$ kernel and stride 2 is applied to a $4 \times 4$ map. What is the output size, and does any output pixel receive more than one tap?

## One heavy pass per image, one light pass per click

A tool that is meant to be clicked on has a budget no classifier has: the answer must come back while the person is still looking at the picture, and the person will click again. That fixes the shape of the model. Whatever can be computed from the image alone must be computed once, and what depends on the click must be as small as it can be made. The result is three parts.

![A two-panel diagram. Left, labelled once per image: the picture goes into the image encoder (the Hiera trunk and neck, 1,812 billion FLOPs), whose output is the image embedding of 256 by 64 by 64 numbers plus two finer maps. Right, labelled once per prompt: the prompts (points, box, mask) go through the prompt encoder to tokens, which enter the mask decoder (3.6 billion FLOPs) together with the stored image embedding, and the decoder outputs masks with an overlap prediction for each.](img/sam-architecture.svg)

The **image encoder** is the vision transformer of Chapter 6 in a large size, run once on a picture of $1{,}024 \times 1{,}024$ pixels. The **prompt encoder** turns points, a box or a rough mask into a few vectors of the same width as the image features. The **mask decoder** reads those vectors and the stored image features and produces masks. The counts in the figure are derived below, and they are the reason the arrangement works: one click costs about 0.2 percent of what the picture cost.

## The encoder that runs once

Brain has two image encoders for this task, and the difference between them is the history of the family. The first model, SAM, uses a plain vision transformer. Its tower in brain, `crates/sam1`, has the geometry of the base size: patches of 16 pixels on a $1{,}024 \times 1{,}024$ picture give a $64 \times 64$ grid of 4,096 tokens of width 768, twelve blocks of 12 heads, and attention that is windowed in eight blocks and global in four (blocks 2, 5, 8 and 11). The windows are $14 \times 14$ tokens, and since 14 does not divide 64 the grid is padded with zeros to $70 \times 70$ and cut into 25 windows. Chapter 6 counted what that mixture saves; the part that is new here is two details of how SAM's blocks work.

The first is position. A ViT adds a learned table of absolute positions once, at the input (resampled to a new grid by the rule of Chapter 6, `crates/sam1/src/resample.rs`), and attention then has to rediscover where things are relative to each other. SAM's blocks add a second, relative term inside attention. For a query at row $q_r$ and column $q_c$ of a window and a key at row $k_r$ and column $k_c$, the score gets an extra number

$$
a^{\mathrm{rel}}_{qk} = \mathbf{q} \cdot \mathbf{U}^{\mathrm{r}}[\,q_r - k_r + M - 1\,] + \mathbf{q} \cdot \mathbf{U}^{\mathrm{c}}[\,q_c - k_c + M - 1\,], \tag{1}
$$

where $M$ is the extent of the span (the window, or the whole grid in a global block) and $\mathbf{U}^{\mathrm{r}}$, $\mathbf{U}^{\mathrm{c}}$ are two learned tables of $2M - 1$ rows, one row for each possible offset along an axis. It is the relative position of Chapter 6's discussion, split into a row part and a column part, and the split is what makes it affordable. A bias that depended on the pair of positions would be one number per head and (query, key) pair. Equation (1) needs only, for each query, its dot product with the rows of two small tables.

<!-- worked: segment_anything.relative_bias -->
seed 0x8894b1a8fb263ec6 (9841686425037586118). Window of 3 by 3 tokens, head width 4. Draws, in this order: the table Rh (5 rows of 4), the table Rw (5 rows of 4), then the query q (4), each entry `round(normal(), 1)`. Rh = [[-0.6, -0.7, 0.3, -0.1], [0.1, -1.1, -1.8, 0.4], [-1.0, 2.5, -1.1, -0.8], [-0.5, 0.2, 0.5, -1.6], [-0.6, -1.2, -1.2, -0.7]], Rw = [[0.1, -0.8, 3.1, 0.2], [-0.1, -1.1, -0.3, -1.4], [-0.9, -1.1, 1.7, -0.4], [-1.1, -1.0, 1.4, 1.1], [-1.7, -1.6, -0.3, -0.6]], q = [1.0, 1.8, 0.3, 0.5]. The query sits at row 1, column 1 of the window. The bias it adds to the score of the key at (kh, kw) is q . Rh[qh - kh + 2] + q . Rw[qw - kw + 2].

The two small vectors, one entry per key row and per key column: rel_h = (-0.79, 2.77, -2.22), rel_w = (-1.93, -2.57, -2.87).

The bias map over the 3 by 3 keys, as the sum rel_h[kh] + rel_w[kw]:

```text
 -2.72   -3.36   -3.66
  0.84    0.20   -0.10
 -4.15   -4.79   -5.09
```

Evaluating the formula pair by pair, 9 times, gives the same map; the largest difference is 0.0.

Storage at the scale of a global block of the first SAM's ViT-B tower (64 by 64 tokens, 12 heads, 4-byte numbers):

| quantity | entries | bytes |
|---|---:|---:|
| bias slab, one number per head and (query, key) pair | 201,326,592 | 805,306,368 |
| two small tables per head and query, (64 + 64) numbers | 6,291,456 | 25,165,824 |
| ratio |  | 32.0 |
| learned tables of the block (127 rows of 64 for each axis, 12 heads share them) | 16,256 | 65,024 |
<!-- /worked -->

For a global block of the base tower the pair-by-pair slab would hold 805,306,368 bytes, and the hoisted form holds 25,165,824, a factor of 32. That is the reason `crates/kernels` has six kernels for this bias (commit 8295a9a10, which added them together with the padded window plan): the contraction with the table is done first into a small intermediate and then folded into the scores in place, so that the large slab is never materialized.

The second detail is the padding, and it is the kind of detail that looks like an implementation choice and is part of the model. The grid is padded after the normalization that precedes attention, so a padded token enters the key and value projections as the zero vector. Its key is then not zero but the projection's bias, and it takes a real share of every softmax in its window. The next table shows the size of the effect on a window of three real tokens and one pad, against the model in which the pad is masked out.

<!-- worked: segment_anything.pad_keys -->
seed 0xabd7bdd88b92148f (12382574438230660239). Draws, in this order: the 6 entries of the three tokens x (row by row) and the 2 of the query q, each `round(normal(), 1)`; Wk (2 x 2) as `round(0.5 * normal(), 1)`, then bk (2) as `round(normal(), 1)`; Wv as Wk, then bv as bk. x = [[-1.4, -0.4], [-0.7, -0.5], [0.3, 0.0]], q = [-1.3, 1.6], Wk = [[-0.1, -0.7], [0.7, 0.7]], bk = [0.6, 0.6], Wv = [[-0.5, -0.6], [-0.2, -0.5]], bv = [-1.7, -0.2]. A key is x Wk + bk, a value x Wv + bv, the score q . k / sqrt(2). The pad token's input is the zero vector, so its key is bk and its value bv.

| window | weights on the 3 real tokens and the pad | output |
|---|---|---|
| masked pad (3 keys) | (0.519, 0.313, 0.168), 0 | (-1.179, 0.520) |
| pad as the layer computes it (4 keys) | (0.430, 0.260, 0.139, 0.171) | (-1.268, 0.396) |
<!-- /worked -->

The weights on the real tokens fall from 0.519, 0.313 and 0.168 to 0.430, 0.260 and 0.139, and the output moves. A checkpoint was trained with the pad in its softmax, so a port that masks it computes a different function, and a port that pads before the norm computes yet another. Brain's tower pads after `norm1` and says why in its module header. The same fact has a second consequence for training: the bias of the key projection receives gradient through the padded positions. The gradient test of commit 8c7739511 found that a mutation dropping exactly those positions from the bias gradient was invisible to the directional gradient check of Volume III and needed a per-entry test, which `crates/sam1/tests/it/gradcheck.rs` now holds, and knowledge #187 records the general lesson: a weight with a use that is easy to miss needs an element-wise check.

The second model, SAM 2, replaces the plain transformer with a hierarchical one, Hiera, which is the pyramid of Chapter 6 with one change that matters for cost: when a stage boundary halves the grid, it does so by max-pooling only the queries of the attention that starts the stage, so the keys keep the old resolution for that block. The trunk of the large model (`crates/sam2/src/config.rs`, `Sam2Config::hiera_large`) has these stages.

<!-- worked: segment_anything.hiera_stages -->
The Hiera trunk of the large SAM 2.1 model (`sam2.1_hiera_l`): patch embedding a 7 by 7 convolution of stride 4 and 3 -> 144 channels on the 1,024 by 1,024 picture; 48 blocks in stages of 2, 6, 36 and 4; the width and the number of heads double, and the grid halves (a 2 by 2 max pool of the queries), at the first block of stages 2, 3 and 4; the windows of the stages are 8, 4, 16 and 8 and a block's window at the first block of a stage is that of the stage before; blocks 23, 33 and 43 are global; an MLP ratio 4. FLOPs count 2 per multiply-add.

| stage | blocks | output grid | width | heads | window(s) | GFLOPs |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 256 x 256 | 144 | 2 | 8 | 70.06 |
| 2 | 6 | 128 x 128 | 288 | 4 | 4, 8 | 212.00 |
| 3 | 36 | 64 x 64 | 576 | 8 | global, 4, 16 | 1,381.15 |
| 4 | 4 | 32 x 32 | 1152 | 16 | 8, 16 | 146.16 |

Patch embedding: 2.77 billion FLOPs. Trunk in all: 1,812.15 billion FLOPs (shortcut projections 16.31, qkv 415.84, attention products 203.09, output projections 130.46, MLPs 1,043.68).
<!-- /worked -->

A $7 \times 7$ convolution of stride 4 cuts the picture into the $256 \times 256$ grid of stage 1, and the position is a $7 \times 7$ learned table resampled to the grid by bicubic interpolation plus a small table tiled over each window. After the last block of each stage the map is stored, and a feature pyramid with $1 \times 1$ lateral convolutions and a top-down path on the two coarsest levels (Chapter 3) turns the stored maps into 256-channel maps. The map at stride 16, $256 \times 64 \times 64$, plus one learned vector, is the **image embedding** the decoder reads. The two finer maps, $32 \times 256 \times 256$ and $64 \times 128 \times 128$, are kept for the end of the decoder, which will want detail that stride 16 has lost.

## Turning a click into tokens

The decoder reads vectors, so a click has to become one. A click is two numbers, and a learned linear map from two numbers to 256 gives a vector that varies smoothly with the position and almost nothing else: Tancik and colleagues showed that a network fed raw coordinates is biased toward low frequencies. What the click vector has to do is make clicks 5 pixels apart look different and clicks 1,000 pixels apart look unrelated, in a way the attention of the decoder can compare with the encoding of the image cells. A code built from sinusoids of the coordinate has that property, and the choice of frequencies is what sets the scale at which positions are told apart.

Take a point $(x, y)$ mapped to $[-1, 1]^2$ and a matrix $\boldsymbol{\Omega}$ of shape $[2, n_\Omega]$ of frequencies, and form

$$
\boldsymbol{\varphi}(x, y) = \bigl[\sin a_1, \dots, \sin a_{n_\Omega},\ \cos a_1, \dots, \cos a_{n_\Omega}\bigr], \qquad a_j = 2\pi\,(x\,\Omega_{0j} + y\,\Omega_{1j}). \tag{2}
$$

```text
function fourier_encode(Omega: array<real>[2, nf], x: real, y: real) -> array<real>[2 * nf]:
    out = zeros(real, [2 * nf])
    for j in 0 .. nf:
        a = 2.0 * pi * (x * Omega[0, j] + y * Omega[1, j])
        out[j] = sin(a)
        out[nf + j] = cos(a)
    return out
```

The entries of $\boldsymbol{\Omega}$ are drawn once from a normal distribution of standard deviation $\sigma_\Omega$ and then frozen (the checkpoint stores the matrix as a tensor of shape $[2, 128]$, `sam_prompt_encoder.pe_layer.positional_encoding_gaussian_matrix`). They are random where Volume V's sinusoidal encoding of Chapter 8 used a fixed geometric ladder, and the reason shows up when two codes are compared. The dot product of the encodings of two points $\mathbf{c}$ and $\mathbf{c}'$ is a sum of products of sines and of cosines, and $\sin a \sin a' + \cos a \cos a' = \cos(a - a')$, so

$$
\boldsymbol{\varphi}(\mathbf{c}) \cdot \boldsymbol{\varphi}(\mathbf{c}') = \sum_{j=1}^{n_\Omega} \cos\bigl(2\pi\, (\mathbf{c} - \mathbf{c}') \cdot \boldsymbol{\Omega}_{:j}\bigr). \tag{3}
$$

Only the difference $\boldsymbol{\delta} = \mathbf{c} - \mathbf{c}'$ appears: the code compares positions the way a relative encoding does, though it is absolute. Now average over the random frequencies. The expected value of $\cos(2\pi\,\boldsymbol\delta \cdot \boldsymbol\omega)$ for a vector $\boldsymbol\omega$ of independent normal entries of standard deviation $\sigma_\Omega$ is the characteristic function of the normal distribution, $e^{-2\pi^2 \sigma_\Omega^2 \lVert\boldsymbol\delta\rVert^2}$, so

$$
\frac{1}{n_\Omega}\,\boldsymbol{\varphi}(\mathbf{c}) \cdot \boldsymbol{\varphi}(\mathbf{c}') \;\approx\; \exp\!\bigl(-2\pi^2 \sigma_\Omega^2\,\lVert\mathbf{c} - \mathbf{c}'\rVert^2\bigr), \tag{4}
$$

a Gaussian kernel whose width is set by $\sigma_\Omega$. A large $\sigma_\Omega$ makes the similarity of two nearby points fall off quickly, so the code can tell apart positions that are close; a small one makes it fall off slowly, a smooth code. The approximation has the accuracy of an average of $n_\Omega$ terms each between $-1$ and $1$, an error near $1/\sqrt{n_\Omega}$ times a number below one.

<!-- worked: segment_anything.fourier_encoding -->
seed 0x75d09aae9528b657 (8489455372210845271). Draws, in this order: the 4 entries of row 0 of G (the x frequencies), then the 4 of row 1 (the y frequencies), each one `normal()` (scale 1). G row 0 = (1.443, 0.029, 3.349, 0.436), row 1 = (0.213, -1.229, -0.235, 1.149).

| point (u, v) in the picture | (x, y) = (2u - 1, 2v - 1) | sin 0 | sin 1 | sin 2 | sin 3 | cos 0 | cos 1 | cos 2 | cos 3 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| (0.3, 0.7) | (-0.4, 0.4) | -0.051 | 0.021 | -0.405 | 0.976 | -0.999 | -1.000 | -0.914 | -0.219 |
| (0.3, 0.8) | (-0.4, 0.6) | -0.314 | 1.000 | -0.121 | -0.093 | -0.950 | -0.005 | -0.993 | -0.996 |
| (0.9, 0.1) | (0.8, -0.8) | -0.102 | 0.042 | -0.740 | 0.427 | 0.995 | 0.999 | 0.672 | -0.904 |

Dot product of the encodings divided by 4: (0.3, 0.7) with (0.3, 0.8) is 0.518, (0.3, 0.7) with (0.9, 0.1) is -0.422, a point with itself is 1.000.
<!-- /worked -->

<!-- worked: segment_anything.fourier_kernel -->
seed 0xaed366ba76c08952 (12597525533727689042). Draws: the 128 entries of row 0 of Z, then the 128 of row 1, each one `normal()`; G = sigma Z for sigma = 1 and 4. The two points are (x, y) = (-0.2, 0.1) and (-0.2 + delta, 0.1), estimate = (phi(p) . phi(p')) / 128 with the 256-channel encoding, exact = exp(-(2 pi sigma delta)^2 / 2) (library exp).

| delta (in the [-1, 1] frame) | sigma 1, estimate | sigma 1, exact | sigma 4, estimate | sigma 4, exact |
|---|---:|---:|---:|---:|
| 0.00 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.05 | 0.960 | 0.952 | 0.519 | 0.454 |
| 0.10 | 0.849 | 0.821 | 0.074 | 0.042 |
| 0.20 | 0.519 | 0.454 | 0.063 | 0.000 |
| 0.30 | 0.229 | 0.169 | 0.050 | 0.000 |
| 0.50 | 0.012 | 0.007 | 0.075 | 0.000 |
<!-- /worked -->

The table shows (4) at work. With $\sigma_\Omega = 1$ two points a tenth of the picture apart ($\delta = 0.2$ in the $[-1, 1]$ frame) still have a similarity of 0.454; with $\sigma_\Omega = 4$ they are already unrelated (0.000 in the formula, 0.063 in the 128-feature sample, which is the noise of the average). Four features, as in the first table, are not enough for the kernel to emerge, and the first table is there to show the arithmetic, not the kernel.

The second ingredient is a type. A click may be a foreground click or a background one, and a box is two clicks that are corners, so each point carries one of four learned embeddings, added to its encoding: background (label 0), foreground (1), top-left corner (2) and bottom-right corner (3). A box is thus the pair of tokens of its corners and goes through exactly the code path of two clicks, which is how `crates/sam2/src/hostpe.rs` handles it (`embed_points`). Brain's `label_slot` panics by name on any other label, because the reference leaves such a point with a bare positional term and no embedding, which is a vector no test covers. Last, a padding point with label $-1$ is always appended: it receives the learned vector `not_a_point` in place of the positional encoding, and it makes the token count the same whether or not a real point is there.

<!-- worked: segment_anything.prompt_tokens -->
one foreground click at (x 2, y 2):

| token | d0 | d1 | d2 | d3 | d4 | d5 | d6 | d7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| click (label 1) | 1.812 | -0.538 | 2.098 | 0.241 | -1.566 | 0.420 | 0.885 | -0.806 |
| padding (label -1) | 0.321 | -0.074 | -0.099 | -0.798 | -0.221 | -0.084 | -0.225 | -0.035 |

a foreground click and a background click:

| token | d0 | d1 | d2 | d3 | d4 | d5 | d6 | d7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| click (label 1) | 1.812 | -0.538 | 2.098 | 0.241 | -1.566 | 0.420 | 0.885 | -0.806 |
| click (label 0) | 1.428 | -2.904 | 0.730 | 1.575 | 0.083 | 0.801 | -0.214 | 1.329 |
| padding | 0.321 | -0.074 | -0.099 | -0.798 | -0.221 | -0.084 | -0.225 | -0.035 |

a box from (x 1, y 1) to (x 4, y 4):

| token | d0 | d1 | d2 | d3 | d4 | d5 | d6 | d7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| corner (label 2) | -0.465 | 0.618 | 0.596 | -0.305 | -0.087 | 2.192 | -0.641 | -2.468 |
| corner (label 3) | -0.635 | -0.115 | 0.077 | 1.531 | 0.655 | 1.571 | 1.244 | 0.182 |
| padding | 0.321 | -0.074 | -0.099 | -0.798 | -0.221 | -0.084 | -0.225 | -0.035 |
<!-- /worked -->

The third prompt type is a rough mask from an earlier round, and it is dense, so it does not join the tokens. It is reduced by two strided convolutions to the size of the image embedding and added to it. When there is no mask, a single learned vector, `no_mask_embed`, is added to every cell instead.

<!-- worked: segment_anything.prompt_shapes -->
The image is 1024 x 1024 pixels and the embedding 64 x 64 (16 pixels per cell).

| stage | output shape | note |
|---|---|---|
| mask prompt | 1 x 256 x 256 | input, a logit map at 4 x the embedding side |
| conv 2 x 2, stride 2, LayerNorm2d, GELU | 4 x 128 x 128 | 1 -> 4 channels |
| conv 2 x 2, stride 2, LayerNorm2d, GELU | 16 x 64 x 64 | 4 -> 16 channels |
| conv 1 x 1 | 256 x 64 x 64 | 16 -> 256 channels, added to the image embedding |
| no mask given | 256 x 64 x 64 | the one learned vector no_mask_embed, copied to every cell |

Weights of the three convolutions (without biases and norms): 4,368.
<!-- /worked -->

## Tokens read the image and the image reads the tokens

The decoder has to turn a handful of prompt tokens and a grid of 4,096 image cells into masks. If it were a transformer decoder in the sense of Volume V it would let the tokens attend to the image, and the tokens would carry the answer out. SAM's decoder does that and also lets the image attend to the tokens, in each layer, and the second direction is what lets the click reach every cell: after the first layer the cells are not the cells of the encoder any more, they are cells that have heard what is being asked.

Start with what goes in. Besides the prompt tokens (the click tokens and the padding token), the decoder has its own learned **output tokens**: one that will predict the overlap, four that will each produce a mask, and in SAM 2 a sixth that predicts whether the object is present at all. They are parameters of the model, with no input; they are in the sequence so that attention can fill them with what the prompt and the image say and so that each of them can be read out at the end. For one click the sequence has $1 + 1 + 4 + 2 = 8$ tokens (the object-score token, the overlap token, four mask tokens, one click and one padding point). The image side is the 4,096 cells of the embedding, with the dense prompt added, and each cell carries a position: the Fourier code of equation (2) at its centre, which the decoder adds to queries and keys but not to values.

![A flow diagram of one two-way layer. On the left the tokens (8 by 256) pass through self-attention, then token-to-image attention whose keys and values come from the image cells, then an MLP. On the right the image cells (4,096 by 256) feed the keys and values of the token-to-image attention and then pass through image-to-token attention whose keys and values are the updated tokens. Both streams end as updated tokens and updated cells.](img/two-way-block.svg)

One layer has four steps in this order. The tokens attend to each other. The tokens attend to the image cells, the cross-attention of Volume V's encoder-decoder chapter with the tokens as queries. An MLP acts on each token. And the cells attend to the tokens, the cross-attention turned around, with the cells as queries and the updated tokens as keys and values. Each step is followed by a residual add and a layer normalization, with one exception that is worth a sentence: in the first layer the token self-attention has neither the position nor the residual, because the tokens that enter are already their own position encoding and there is nothing earlier to add to. As an algorithm, with `attend` the single-head attention of Volume V (the weights `W.sa`, `W.t2i` and `W.i2t` each hold the four matrices of projections $\mathbf{W}_q, \mathbf{W}_k, \mathbf{W}_v, \mathbf{W}_o$), and `layer_norm_rows` the normalization of each row to mean 0 and variance 1:

```text
function attend(Xq: array<real>[Nq, D], Xk: array<real>[Nk, D], Xv: array<real>[Nk, D],
                Wq: array<real>[D, A], Wk: array<real>[D, A], Wv: array<real>[D, A], Wo: array<real>[A, D]) -> array<real>[Nq, D]:
    Q = matmul(Xq, Wq)
    K = matmul(Xk, Wk)
    V = matmul(Xv, Wv)
    S = matmul(Q, transpose(K)) / sqrt(A)
    return matmul(matmul(softmax_rows(S), V), Wo)

function two_way_layer(queries: array<real>[T, D], keys: array<real>[N, D], tok_pe: array<real>[T, D],
                       key_pe: array<real>[N, D], first: bool, W) -> (array<real>[T, D], array<real>[N, D]):
    if first:
        queries = attend(queries, queries, queries, W.sa.q, W.sa.k, W.sa.v, W.sa.o)
    else:
        q = queries + tok_pe
        queries = queries + attend(q, q, queries, W.sa.q, W.sa.k, W.sa.v, W.sa.o)
    queries = layer_norm_rows(queries)
    o = attend(queries + tok_pe, keys + key_pe, keys, W.t2i.q, W.t2i.k, W.t2i.v, W.t2i.o)
    queries = layer_norm_rows(queries + o)
    h = matmul(map(relu, matmul(queries, W.up)), W.down)
    queries = layer_norm_rows(queries + h)
    o = attend(keys + key_pe, queries + tok_pe, queries, W.i2t.q, W.i2t.k, W.i2t.v, W.i2t.o)
    keys = layer_norm_rows(keys + o)
    return (queries, keys)
```

Positions enter only through the queries and the keys, never the values: `queries + tok_pe` and `keys + key_pe` are what are compared, and `keys` and `queries` alone are what is copied. The model is thus free to match on position without moving a positional code into the content. Two layers are stacked, and one more token-to-image attention with a residual and a normalization follows, so the tokens read the final, updated cells once more. In the large model the cross-attentions use 8 heads of width 16 inside a projection to half the model width, 128, and the self-attention uses the full 256; the MLP is 256 to 2,048 and back.

To see the shapes and the arithmetic without 256-wide matrices, here is the whole decoder with width 8, one head, one upscaling and a $4 \times 4$ image embedding on an $8 \times 8$ picture, all weights random. Every tensor and its scale is listed so that the draw order can be repeated.

<!-- worked: segment_anything.toy_setup -->
seed 0x352233ac9aa94763 (3828679449640519523). One generator draws every tensor, each entry one `normal()` times the scale shown, tensors in the order of the table, each row-major (a weight [in, out] is drawn row by row; `y = x W`). All biases are 0, all LayerNorm gains 1 and offsets 0 (epsilon 1e-5).

| tensor (draw order, top to bottom) | shape | scale |
|---|---|---|
| G, the Fourier matrix | 2 x 4 | 1 |
| point_emb, the 4 type embeddings | 4 x 8 | 1 |
| not_a_point | 8 | 1 |
| iou_token | 8 | 1 |
| mask_tokens | 4 x 8 | 1 |
| no_mask, the dense embedding of no mask | 8 | 1 |
| image embedding, 4 x 4 tokens (row-major), each `round(normal(), 1)` | 16 x 8 | 1 |
| per layer (2 layers): self-attention q, k, v, out | 4 x (8 x 8) | 1/sqrt(8) |
| per layer: token-to-image q, k, v, out | 3 x (8 x 4), 4 x 8 | 1/sqrt(fan-in) |
| per layer: MLP up, down | 8 x 16, 16 x 8 | 1/sqrt(fan-in) |
| per layer: image-to-token q, k, v, out | 3 x (8 x 4), 4 x 8 | 1/sqrt(fan-in) |
| final token-to-image q, k, v, out | 3 x (8 x 4), 4 x 8 | 1/sqrt(fan-in) |
| transposed convolution W[c][c'][a][b] | 8 x 2 x 2 x 2 | 1/sqrt(8) |
| hypernetwork of mask token i = 0..3: layer 1, layer 2 | 8 x 8, 8 x 2 | 1/sqrt(fan-in) |
| IoU head: layer 1, layer 2 | 8 x 8, 8 x 4 | 1/sqrt(fan-in) |
<!-- /worked -->

<!-- worked: segment_anything.decoder_shapes -->
The prompt is the one foreground click of the previous tensors. Shapes (rows x columns) through the decoder.

| quantity | shape |
|---|---|
| tokens: iou, 4 masks, click, padding | 7 x 8 |
| image tokens (4 x 4 cells) with position | 16 x 8 |
| token-to-image weights, one layer | 7 x 16 |
| image-to-token weights, one layer | 16 x 7 |
| tokens leaving the transformer, hs | 7 x 8 |
| upscaled image features | 2 x 8 x 8 |
| hypernetwork outputs, one per mask token | 4 x 2 |
| mask logits | 4 x 64 (= 4 maps of 8 x 8) |
| predicted overlaps | 4 |
<!-- /worked -->

The token-to-image attention of the click token shows what the cross-attention does. Each row of a weight matrix sums to 1, and the weights of one token are a distribution over the 16 cells. With random projections nothing makes the weights concentrate near the click, and they do not; a trained model has weights that do, and the toy is there to show where in the computation they act.

<!-- worked: segment_anything.decoder_attention -->
token-to-image, layer 0, the click token's weights over the 4 x 4 cells (row-major; each row of the map sums to 1):

```text
  0.125    0.039    0.007    0.024
  0.159    0.006    0.040    0.013
  0.012    0.075    0.016    0.015
  0.163    0.188    0.059    0.061
```

sum = 1.000000

token-to-image, layer 1, the click token's weights over the 4 x 4 cells (row-major; each row of the map sums to 1):

```text
  0.069    0.183    0.054    0.017
  0.131    0.080    0.019    0.015
  0.111    0.061    0.030    0.051
  0.038    0.025    0.029    0.087
```

sum = 1.000000

final token-to-image, the click token:

```text
  0.021    0.457    0.162    0.004
  0.003    0.029    0.008    0.001
  0.155    0.111    0.010    0.006
  0.028    0.004    0.001    0.001
```

image-to-token, layer 0: the weight that each of the 16 cells gives to the click token (each cell's row over the 7 tokens sums to 1; the column is not normalized):

```text
  0.332    0.182    0.024    0.344
  0.280    0.199    0.189    0.346
  0.182    0.205    0.401    0.416
  0.076    0.400    0.389    0.427
```
<!-- /worked -->

The last column of the image-to-token attention is a different object: for each cell, the weight it gives to the click token among the seven. It is not a distribution over cells, because each cell's row is normalized across the tokens. It is the transposed question, and it is what lets the click's influence reach every cell.

## From tokens to a mask

The tokens have read the image. Now they must be turned into masks, and the decision that makes this cheap is that a mask is not decoded token by token into pixels; it is the result of one dot product per pixel.

The updated cells are reshaped back into a map of the embedding's size and enlarged by transposed convolutions of kernel 2 and stride 2 (Chapter 2), each of which doubles the sides: from $64 \times 64$ to $128 \times 128$ and then to $256 \times 256$, with channel counts 256, 64 and 32. In SAM 2 the finer maps of the encoder, $64 \times 128 \times 128$ and $32 \times 256 \times 256$, are added after each transposed convolution, which is where the detail stride 16 lost comes back. Call the final map $\mathsf{F}$, of $C'$ channels. Then each of the four mask tokens goes through its own small MLP, a hypernetwork, whose output is a vector $\mathbf{h}_i$ of $C'$ numbers, and the logit of mask $i$ at pixel $p$ is

$$
\ell_i(p) = \mathbf{h}_i \cdot \mathsf{F}[:, p], \tag{5}
$$

a dot product per pixel, which for all four masks is one matrix product of $[4, C']$ by $[C', P]$. Each mask token has become the weights of a tiny linear classifier over the pixel features, and it was written by the attention in the layers before. The same four vectors would be useless for a different prompt; they are made fresh for each. That is why the cost per click is what it is: the upscaled map $\mathsf{F}$ depends on the click only through the cells, and the pixel loop is a matrix product.

<!-- worked: segment_anything.decoder_masks -->
mask token 0, logits at the 8 x 8 output positions:

```text
   2.15     1.48     1.91     1.39     1.48     0.65     1.98     1.54
   0.81    -0.19     0.85     0.01     1.45    -0.32     1.72     0.25
   2.22     2.06     2.14     1.79     2.02     1.73     2.09     1.78
   1.15     0.66     1.00     0.50     1.16    -0.17     1.18    -0.04
   2.22     1.50     2.23     1.95     2.13     1.68     2.09     1.75
   0.68    -0.22     1.08     0.12     1.14     0.19     1.30     0.09
   2.00     1.05     2.10     1.93     2.12     1.55     1.87     1.30
   0.78    -0.29     1.51     0.77     1.05     0.06     1.46     0.72
```

mask token 1, logits at the 8 x 8 output positions:

```text
  -3.60    -0.61    -3.21    -1.10    -2.58    -0.25    -3.33    -0.76
   0.58    -0.06    -0.06    -0.26    -1.68     0.19    -0.70    -0.67
  -3.57    -1.32    -3.54    -1.24    -3.40    -0.93    -3.49    -0.93
   0.45    -1.30    -0.19    -1.02     0.17    -0.06    -0.47    -0.23
  -3.70    -0.47    -3.71    -1.01    -3.57    -0.99    -3.49    -1.00
   0.12    -0.01     0.34    -0.50    -0.00    -0.59    -0.43    -0.42
  -3.37    -0.26    -3.52    -1.35    -3.54    -0.81    -3.08    -0.97
   0.09     0.09    -0.28    -1.40     0.07    -0.41    -0.29    -1.34
```

mask token 2, logits at the 8 x 8 output positions:

```text
  -0.95     0.66    -0.85     0.25    -0.73     0.31    -0.89     0.60
   1.01    -0.18     0.59    -0.18    -0.11    -0.10     0.78    -0.29
  -0.88     0.60    -0.92     0.45    -0.91     0.62    -0.92     0.67
   1.16    -0.43     0.61    -0.35     0.98    -0.16     0.54    -0.20
  -0.97     0.78    -0.98     0.73    -0.95     0.55    -0.93     0.59
   0.59    -0.17     1.04    -0.27     0.84    -0.28     0.65    -0.23
  -0.90     0.60    -0.93     0.47    -0.93     0.58    -0.79     0.27
   0.64    -0.15     0.92    -0.42     0.83    -0.25     0.88    -0.41
```

mask token 3, logits at the 8 x 8 output positions:

```text
   0.37     0.75     0.32     0.56     0.22     0.34     0.33     0.74
   0.65    -0.13     0.51    -0.06     0.44    -0.14     0.87    -0.02
   0.42     0.92     0.37     0.77     0.34     0.81     0.36     0.85
   0.82     0.06     0.56     0.04     0.76    -0.12     0.60    -0.09
   0.38     0.80     0.38     0.93     0.36     0.77     0.35     0.81
   0.45    -0.14     0.75    -0.06     0.70    -0.04     0.68    -0.06
   0.33     0.58     0.36     0.83     0.36     0.74     0.33     0.54
   0.50    -0.15     0.85     0.10     0.66    -0.07     0.82     0.09
```

The logit of mask token 1 at pixel (row 3, column 5) is the dot product of its hypernetwork output (0.6409, -1.7833) with the upscaled features at that pixel (-0.1538, -0.0226): -0.0986 + 0.0403 = -0.0583; the table above has -0.0583.

Predicted overlaps of the four mask tokens (sigmoid of the IoU head): (0.3503, 0.3709, 0.2464, 0.4326). With several clicks the model uses token 0 alone; with one click it returns tokens 1, 2 and 3, and the one with the highest predicted overlap is token 3 here.
<!-- /worked -->

The table checks (5) on one pixel, and the four maps show what the machinery produces with random weights: four different maps from one prompt, as many as there are mask tokens, with no meaning attached. The overlap head is another small MLP, on the overlap token, with a sigmoid, giving one number in $(0, 1)$ per mask. Which tokens are returned depends on how many clicks there were: with several clicks the ambiguity is mostly gone and only token 0 is used; with one click, or when the caller wants the ambiguity, tokens 1, 2 and 3 are returned, and the one with the highest predicted overlap is the default choice (`prompt.multimask_output` in `crates/sam2/src/model.rs`).

The same weights answer different prompts differently, because the prompt enters twice, in the tokens and in the keys the cells are compared with.

<!-- worked: segment_anything.decoder_prompts -->
The same weights and the same image embedding; only the prompt changes. Mask token 1.

| prompt | tokens | pixels with logit > 0 | largest difference of logit from the one-click map |
|---|---:|---:|---:|
| click | 7 | 9 | 0.000 |
| click and background click | 8 | 5 | 0.782 |
| box | 8 | 2 | 0.977 |
<!-- /worked -->

SAM 2 adds a sixth token that predicts an **object score**, the logit of the statement "the prompted object is visible in this image". If the score is not positive, every mask logit is replaced by $-1024$ and the object pointer, defined below, is replaced by a learned one. For a still image this is a safety valve; in a video it is the occlusion signal, and it gets its own section.

Training has the three parts promised at the start. The mask loss is the sum of a focal loss and a dice loss on the logits, with weights 20 and 1 in the reference recipe, and it is charged to the mask with the smallest such loss among those returned, and only to that one. The overlap head is trained with a squared error against the overlap each returned mask actually has with the truth. The choice of the best mask is not differentiated: it is a piecewise constant function of the weights, so brain freezes it as a target (`crates/sam2/src/train.rs`, the `MaskTargets` type, whose `mask_w` weights select the channel) before it takes gradients, and the finite-difference checks of Volume III can then run on a smooth objective. In the toy of the opening, the rule has a plain reading: for the reading "the core", the best candidate is tolerance 0, for "the whole object" it is tolerance 3, and each reading trains the head that is closest to it.

## What a click costs

The claim of the first section was that the picture is expensive and the click is cheap. Count it, with 2 FLOPs per multiply-add, the Hiera trunk from the table above, the ViT-B tower of `crates/sam1` with its neck for comparison (the padded grid for windowed blocks, global blocks on 4,096 tokens, relative-position terms not counted), and the decoder with the sizes of the large model.

<!-- worked: segment_anything.encoder_vs_decoder -->
FLOPs count 2 per multiply-add; norms, activations and the relative-position terms are not counted.

| quantity | GFLOPs | relative to the Hiera-L trunk |
|---|---:|---:|
| Hiera-L trunk (SAM 2.1), once per image | 1,812.15 | 1.00 |
| ViT-B tower with neck (the SAM 1 tower of brain), once per image | 967.17 | 0.53 |
| mask decoder of SAM 2, one click, once per prompt | 3.64 | 0.0020 |

The decoder by part (one click: 1 point token and the padding token, 4 mask tokens, the IoU token and the object-score token, 8 tokens in all):

| part | MFLOPs | share (%) |
|---|---:|---:|
| transformer | 2,815.6 | 77.4 |
| upscaling | 805.3 | 22.1 |
| hypernetworks | 1.1 | 0.0 |
| mask dot products | 16.8 | 0.5 |
| heads | 0.5 | 0.0 |

The ViT-B tower pads its 64 by 64 grid to 70 by 70 for windows of 14 (4,900 rows in 25 windows, against 4,096). Features the decoder needs from one image, in 32-bit numbers: the image embedding 256 x 64 x 64, the two high-resolution maps 32 x 256 x 256 and 64 x 128 x 128: 16,777,216 bytes (16 MiB).

| prompts on one image | work (GFLOPs), encoder once | work (GFLOPs), encoder per prompt |
|---|---:|---:|
| 1 | 1,815.79 | 1,815.79 |
| 2 | 1,819.43 | 3,631.57 |
| 8 | 1,841.26 | 14,526.29 |
<!-- /worked -->

A click is 3.64 billion FLOPs against 1,812 for the picture, and the encoder's share of the work of one picture and one click is 99.8 percent. Inside the decoder, three quarters of the arithmetic is in the transformer, and most of that is not the tokens at all: it is the projections of the 4,096 image cells to keys and values (in the token-to-image attention) and to queries (in the image-to-token attention), $2 \cdot 4{,}096 \cdot 256 \cdot 128$ FLOPs each, in every layer. The tokens are eight rows and cost almost nothing. A decoder that did not touch the cells would be cheaper and would be a different model, since the cells' attention to the tokens is what spreads the click through the map. The 16 MiB of features per image is what an interface has to hold between clicks. Brain's serving provider keeps exactly this: `crates/sam2/src/caps.rs` caches the last encoder output keyed by a hash of the image, so a second prompt on the same image skips the trunk, and the residency adapter groups a batch of requests by image so that $n$ prompts on one image cost one trunk pass (`crates/catalog/src/resident_sam2.rs`).

## Remembering the last frames

A video is a sequence of images, and the model so far would handle it by being told, on every frame, which object to segment. What is wanted is that a click on frame 0 is enough. The object moves, changes shape, is partly hidden and comes back, and the model has to follow it. SAM 2 does this by giving the decoder a memory.

The idea has three steps. After a frame has been segmented, its mask is **encoded into a memory**: a small feature map that says what the object looked like and where it was. The memories of recent frames, and of the frames where the user prompted, are kept in a **memory bank**. And before the decoder runs on a new frame, the frame's features attend to the bank, so that the cells that look like the earlier object, in about the earlier place, are marked as such. That last step is **memory attention**, and its output replaces the image embedding in the decoder; the prompt encoder and the decoder are otherwise the ones of the image path (in `crates/sam2/src/model.rs`, `decode_with` takes the feature map as an argument for exactly this reason).

The memory encoder is a small convolutional network. The mask logits are turned into $20\,\sigma(\ell) - 10$, which maps the sigmoid to $(-10, 10)$.

<!-- worked: segment_anything.mask_to_memory -->
A mask logit enters the memory encoder as 20 * sigmoid(logit) - 10 (library exp).

| mask logit | sigmoid | value given to the memory encoder |
|---|---:|---:|
| -10.0 | 0.0000 | -9.999 |
| -4.0 | 0.0180 | -9.640 |
| -1.0 | 0.2689 | -4.621 |
| 0.0 | 0.5000 | 0.000 |
| 1.0 | 0.7311 | 4.621 |
| 4.0 | 0.9820 | 9.640 |
| 10.0 | 1.0000 | 9.999 |
<!-- /worked -->

The values are squeezed to the grid of the embedding by four convolutions of kernel 3 and stride 2 (channels 1, 4, 16, 64, 256), added to a $1 \times 1$ projection of the frame's own top-level feature map, mixed by two ConvNeXt blocks (a depthwise $7 \times 7$ convolution of Chapter 2, a normalization and two pointwise layers) and reduced to 64 channels. The result, $64 \times 64 \times 64$ for a clip at full size, is one memory slab.

Which slabs does frame $t$ read? The bank keeps, for every frame it has seen, its slab and its **object pointer**: a 256-vector computed by an MLP from the output token of the best mask, which summarizes the object as the decoder saw it. The frame reads the slab of each prompted frame, always, and the slabs of the six frames before it, so seven slabs in all, each labelled by a temporal position, a **slot** from 0 to 6 (the prompted frame is slot 0 and the previous frame is slot 6). It also reads up to 16 pointers, each cut into $256/64 = 4$ tokens of width 64 to match the slabs, and each marked with its distance in frames from the current one through a sinusoid projected to 64 numbers. The rule has some edge cases at the start of a clip, which the following function writes out (the frame before the current one takes slot $n_{\mathrm{mem}} - 1$; frames that precede the clip or are themselves prompted frames contribute no recent slab):

```text
function memory_slots(frame: int, cond: list<int>, n_mem: int) -> list<(int, int)>:
    slots = []
    for c in cond:
        append (0, c) to slots
    for t_pos in 1 .. n_mem:
        src = frame - (n_mem - t_pos)
        is_prompted = any([src == c for c in cond])
        if src >= 0 and not is_prompted:
            append (t_pos, src) to slots
    return slots
```

<!-- worked: segment_anything.memory_selection -->
A clip of 64 frames, one prompt on frame 0 (the only conditioning frame), the large model: 7 memory slots, 16 object pointers at most, 4,096 cells per slab (a 64 by 64 grid) and 4 tokens per pointer (256 / 64).

| frame t | source frames of the slabs | temporal positions | pointers | keys of the cross-attention |
|---|---|---|---:|---:|
| 1 | 0 | 0 | 1 | 4,100 |
| 2 | 0, 1 | 0, 6 | 2 | 8,200 |
| 3 | 0, 1, 2 | 0, 5, 6 | 3 | 12,300 |
| 7 | 0, 1, 2, 3, 4, 5, 6 | 0, 1, 2, 3, 4, 5, 6 | 7 | 28,700 |
| 8 | 0, 2, 3, 4, 5, 6, 7 | 0, 1, 2, 3, 4, 5, 6 | 8 | 28,704 |
| 12 | 0, 6, 7, 8, 9, 10, 11 | 0, 1, 2, 3, 4, 5, 6 | 12 | 28,720 |
| 20 | 0, 14, 15, 16, 17, 18, 19 | 0, 1, 2, 3, 4, 5, 6 | 16 | 28,736 |
<!-- /worked -->

Frame 1 sees only the prompted frame. From frame 7 on the bank is full: seven slabs of 4,096 cells and one pointer from each of up to 16 frames. At frame 20 the cross-attention has $7 \cdot 4{,}096 + 16 \cdot 4 = 28{,}736$ keys, the number that brain's `MEM_ATTN_CHUNK` comment quotes, and it is the number that is held constant from then on: the cost per frame does not grow with the length of the clip, which is what the bounded bank is for.

![A row of thirteen frames, 0 to 12. Frame 0 is marked as the prompted frame in slot 0, frames 1 to 5 are marked as pointer only, frames 6 to 11 occupy slots 1 to 6, and frame 12 is the current frame. Dashed lines from frame 0 and frames 6 to 11 lead to a box of memory tokens, seven slabs of 4,096 plus twelve pointers of 4 tokens, 28,720 keys, which feeds a box labelled memory attention whose queries are the 4,096 cells of frame 12.](img/memory-bank.svg)

Memory attention itself is four layers, each a self-attention of the current frame's cells, a cross-attention into the memory tokens and an MLP, with the normalization first (a pre-norm layer, as in Volume V). The queries are the 4,096 cells of the current frame, the keys are the memory plus a position code (a sine code of the memory grid and, added to it, a learned vector per slot) and the values are the memory without it. The attention is a single head of width 256, and the keys and values are projected up from 64 to 256, so the cross-attention is one softmax over 28,736 keys for each of 4,096 queries.

Position needs care here, and the reason is concrete. A cell at the top left of the current frame should be compared with the memory around the top left, and the memory is seven slabs of the *same* grid laid end to end. Memory attention rotates queries and keys with the axial rotary embedding of Volume V (Chapter 8's multi-axis RoPE): the first half of the pairs of channels turn by the column and the second half by the row, with base 10,000. The query's table of rotations is repeated down the keys, one copy per slab, because each slab is the same grid from a different frame; and the pointer tokens at the end are not rotated at all, because they have no place in the grid. The rotation makes the score depend on where a key lies *relative to* the query, so memory at the same cell counts for more than memory three cells away. Here is the effect on a grid of $3 \times 3$ cells where an object, shown by its appearance and by a mask channel in the memory, has moved one cell to the right.

<!-- worked: segment_anything.memory_toy -->
A 3 by 3 grid, token t at column x = t mod 3 and row y = t div 3. A cell's feature has width 8: (a, 0, m, 0, a, 0, 0, 0) with the appearance a = 1.4 on the object and -1.4 elsewhere in both the x and y channels, and the mask channel m = 2 on the memory cell that the earlier mask covered and 0 everywhere else. Memory: the object (and its mask) at token 4, column 1 and row 1. Current frame: the object has moved to token 5 (column 2, row 1), and the current cells carry no mask channel. The query, key and value projections are the identity, the score is q . k / sqrt(8), and the rotation (when used) is applied to queries and keys by the cell's (x, y) with base 10000; the values are the unrotated memory features.

Attention without position (content only). Weights of the query at the object's new cell (token 5), over the 9 memory cells:

```text
  0.042    0.042    0.042
  0.042    0.667    0.042
  0.042    0.042    0.042
```

The mask channel read back at each of the 9 current cells:

```text
  0.016    0.016    0.016
  0.016    0.016    1.333
  0.016    0.016    0.016
```

Attention with the axial rotary embedding on queries and keys. Weights of the query at the object's new cell (token 5), over the 9 memory cells:

```text
  0.126    0.065    0.047
  0.091    0.399    0.034
  0.126    0.065    0.047
```

The mask channel read back at each of the 9 current cells:

```text
  0.059    0.038    0.059
  0.038    0.025    0.797
  0.059    0.038    0.059
```
<!-- /worked -->

Without position the query on the object's new cell weights the object's old cell at 0.667 (the other eight cells share the rest) and the mask channel read back at the new cell is 1.333, against 0.016 everywhere else: the mask knowledge of frame $t-1$ has been carried to the cell where the object now is, which is exactly the function of memory attention. With the rotation the weight on the old cell falls to 0.399 and the read-back to 0.797, because the old cell is one column away; a mild preference for staying near where the object was, which a model can use or learn to ignore. The cost of the position code is that an object that moves much farther than one cell between frames has to be found by appearance against a prior of locality, and the two are balanced by training.

An object is not always visible. If the object score of a frame is not positive, the best mask is not used as a mask: its logits are set to $-1024$, the pointer is replaced by the learned `no_obj_ptr`, and the memory encoder adds a learned vector, `no_obj_embed_spatial`, to every cell of the slab it writes, so the bank records "absent" and does not carry a stale mask.

<!-- worked: segment_anything.occlusion_rules -->
| object score | mask | object pointer | memory features |
|---|---|---|---|
| object score > 0 | the best mask's logits | the pointer projected from the output token | none |
| object score <= 0 | every logit replaced by -1024 | the learned no_obj_ptr | the learned no_obj_embed_spatial added to every cell |

A logit of -1024 gives the probability sigmoid(-1024) = 0 (the stable form of the sigmoid, `exp(z) / (1 + exp(z))` for negative z, underflows to 0), and 20 * 0 - 10 = -10 in the memory: the same value as the most confident background.
<!-- /worked -->

When the object comes back, the frames after it find an appearance in the older slabs (the prompted frame is always there) and the pointers, and the score becomes positive again. The record in brain's tracking tests is concrete: commit 6bcc0e79f, which ported the memory bank, found that the occlusion branch was dead code in its own test. The first fixture clip never hid the subject, so a mutation that deleted the `no_obj_embed_spatial` branch survived every gate. The clip now hides the subject on two frames and `crates/sam2/tests/it/video_parity.rs` asserts that the fixture contains a frame the reference calls absent and that the memory of such a frame differs by the learned vector. A gate that cannot fail on the branch is not a gate on the branch.

## What a tracked frame costs

A tracked frame runs the trunk again, since the picture is new, and adds memory attention, the decoder with the tracking prompt (no click: the padding point twice) and the memory encoder.

<!-- worked: segment_anything.frame_costs -->
GFLOPs for one tracked frame of the large model (2 per multiply-add). The decoder runs with the tracking prompt: no click, the padding point twice (8 tokens, as for one click). Memory attention: 4 layers on 4,096 cells, a single head of width 256 whose cross-attention keys and values come from 64-wide memory tokens; the memory encoder: a mask downsampler of 4 stride-2 convolutions (1, 4, 16, 64, 256 channels, kernel 3), two 1 x 1 convolutions, two ConvNeXt blocks (depthwise 7 x 7 and two pointwise layers of ratio 4) and a 1 x 1 convolution to 64.

| frame | memory keys | trunk | memory attention | decoder | memory encoder | total | memory attention (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4,100 | 1,812.15 | 185.83 | 3.64 | 11.61 | 2,013.22 | 9.2 |
| 2 | 8,200 | 1,812.15 | 255.69 | 3.64 | 11.61 | 2,083.08 | 12.3 |
| 8 | 28,704 | 1,812.15 | 605.06 | 3.64 | 11.61 | 2,432.46 | 24.9 |
| 20 | 28,736 | 1,812.15 | 605.61 | 3.64 | 11.61 | 2,433.00 | 24.9 |
<!-- /worked -->

Once the bank is full, memory attention is 605.6 billion FLOPs, 24.9 percent of the frame, and the frame costs a third more than a still image with one click (2,433 against 1,816 billion FLOPs). Its work is dominated by the cross-attention: the scores and averages are $4 \cdot 4{,}096 \cdot 28{,}736 \cdot 256$ FLOPs in each of four layers, 482 billion FLOPs, because the head is a single head of width 256 over every key of every slab. Frame 1, with a one-slab bank, is 9.2 percent. A comment in brain's `Tracker` documentation says the trunk is about 99 percent of the work and that is true of the image path, where it is 99.8 percent, and not of a tracked frame with a full bank, where the count above gives the trunk 74.5 percent of the frame. Which of the two numbers matters depends on the question: the cache that saves a second prompt saves the trunk, and the trunk is what a tracker must run on every frame.

## Where the two models live in brain

The two crates are split in a way that does not match the family. `crates/sam1` holds only the encoder tower: the patch embedding, twelve windowed and global blocks with the relative-position bias, the neck, and a compressor of two strided convolutions that the document reader of Chapter 10 consumes. It has no prompt encoder and no mask decoder, so the first SAM's promptable half is described in this chapter from the paper's design and from SAM 2's code, which keeps the same structure (two-way layers, output tokens, hypernetwork masks) and adds the object score and the high-resolution skips. `crates/sam2` holds everything of the second model: Hiera and the neck (`model.rs`), the prompt tables (`hostpe.rs`), the decoder, the memory bank and tracker (`video.rs`), a mask-sequence file format for handing the tracked masks to another model (`maskseq.rs`), decoder training with the trunk frozen (`train.rs`) and the capability provider (`caps.rs`).

Three facts about how they are verified are worth carrying out of the chapter. First, the forward of each is compared with the reference's own activations at every stage: the SAM 2 image path against dumped taps of the released model for both variants, each tap held to a cosine of at least 0.9999 (`FLOOR` in `crates/sam2/tests/it/parity.rs`; commit fba93e5eb, the first port, reports 283 stage comparisons), and the video path against dumps of a clip with cosine and relative L2 both asserted, because the test's own comments note that a cosine alone cannot see a small systematic error. Second, the import is a gate: `Scope::Video` allows no skipped tensor and errors, naming it, on a checkpoint key with no home or a home with no key, so no module can silently stay at its initial values. Third, the decomposed relative-position kernels and the padded window plan were built into the shared vision code of `crates/model` and not into the tower, and `crates/sam2` reuses them (the Hiera blocks run on the window plan, the decoder's cross-attention on the shared cross-attention kernels), because the point of the engine is that a new model is composed from shared pieces.

## Check your understanding

1. **Explain.** In the opening, the tolerance-6 candidate scores 0.727 against the whole object although it contains it. How can a mask that contains the truth have an overlap below 1, and what in the decoder is there to tell the user?
2. **Find the error.** "A random Fourier encoding of a point is smooth in the point's position because sines are smooth, so two nearby points always get nearly the same code." What is wrong, and which quantity controls it?
3. **Predict.** The padding point (label $-1$) is dropped from every prompt. Which part of the decoder sees a different sequence, and does a prompt without any real point still produce a mask?
4. **Explain.** Why are the keys of the token-to-image attention the image cells plus their positions, while the values are the cells alone?
5. **Find the error.** "Because the click costs 0.2 percent of the picture, a model that re-runs the encoder on every click spends 0.2 percent more." Give the right ratio for 8 clicks on one image.
6. **Predict.** A clip has 200 frames and one prompted frame. The memory bank keeps every frame's slab. How does the number of keys of memory attention at frame 100 compare with the bank of this chapter, and what does it do to the cost of that frame?
7. **Explain.** What would be lost if object pointers were left out and only slabs were kept? What would be lost if slabs were left out and only pointers kept?

## Summary

A promptable segmenter takes an image and a hint and returns a mask. One point leaves the extent of the object open, so the model returns several masks with a predicted overlap each, and is trained on the best of them. The architecture splits into an image encoder that runs once (a windowed and global vision transformer in the first model, a Hiera pyramid in the second), a prompt encoder that turns points, corners and masks into tokens with random Fourier features of the coordinates and learned types, and a light decoder whose tokens and image cells attend to each other in both directions and whose masks are dot products between a token's hypernetwork output and the upscaled cells. Counted, the picture costs 1,812 billion FLOPs and a click 3.64. For video, each segmented frame is encoded into a memory, a bank keeps seven slabs and sixteen pointers, and memory attention marks the current frame's cells by what the object looked like before, with an object score that lets the bank record that the object is gone. A tracked frame costs a third more than a still (2,433 billion FLOPs against 1,816), mostly because of that attention.

The limits: the toy decoder has random weights and shows shapes and arithmetic, not segmentation; the counts are FLOPs and bytes, not seconds; the large model's memory attention, one head over every slab, is a design in which the cost is paid for every frame whether or not the object moved. The next chapter turns from where an object is to what it is called, by learning a space where images and sentences land close together; the vision transformer of Chapter 6 reappears as the image tower of that space.

## References

- A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo, P. Dollar and R. Girshick, *Segment anything*, ICCV 2023. The promptable task, the encoder-prompt-decoder split, the two-way decoder, multiple masks with a predicted overlap and the minimum-loss training.
- N. Ravi, V. Gabeur, Y.-T. Hu, R. Hu, C. Ryali, T. Ma, H. Khedr, R. Radle, C. Rolland, L. Gustafson and others, *SAM 2: segment anything in images and videos*, 2024. The memory encoder, the memory bank, memory attention and the object pointers and score.
- M. Tancik, P. P. Srinivasan, B. Mildenhall, S. Fridovich-Keil, N. Raghavan, U. Singhal, R. Ramamoorthi, J. T. Barron and R. Ng, *Fourier features let networks learn high frequency functions in low dimensional domains*, NeurIPS 2020. Random Fourier features of a low-dimensional input and the role of their scale.
- C. Ryali, Y.-T. Hu, D. Bolya, C. Wei, H. Fan, P.-Y. Huang, V. Aggarwal, A. Chowdhury, O. Poursaeed, J. Hoffman and others, *Hiera: a hierarchical vision transformer without the bells-and-whistles*, ICML 2023. The pyramid used as the trunk of the second model.

## Exercises

1. **[Calculate]** For $\sigma_\Omega = 2$, give the similarity (4) of two points at a distance of 0.1 and of 0.25 in the $[-1, 1]$ frame. At what distance does it fall to one half?
2. **[Calculate]** How many tokens does the large-model decoder hold for a box prompt, and how many for a foreground click plus a background click? Give the shape of one head's weight matrix in the first token-to-image attention in each case.
3. **[Calculate]** For a global block of the base ViT tower with 12 heads, give the bytes of the pair-by-pair relative-position slab and of the hoisted tables for a grid of $48 \times 48$ and of $96 \times 96$ tokens, and the ratio.
4. **[Calculate]** The padded grid of a windowed block has $70 \times 70$ tokens. What share of its qkv products is spent on padding?
5. **[Calculate]** For the large model at frame 3, give the number of keys of the cross-attention and the FLOPs of its score and average products over the four layers.
6. **[Implement]** Reproduce `segment_anything.toy_prompts` and `segment_anything.toy_readings`: the three grown masks and the masked-out neighbour, and the overlap of each candidate with each reading.
7. **[Implement]** Reproduce `segment_anything.fourier_encoding` and `segment_anything.fourier_kernel`: the generator, the draw order and the estimate and the exact kernel of (4).
8. **[Implement]** Reproduce `segment_anything.toy_setup`, `segment_anything.decoder_masks` and `segment_anything.decoder_prompts`: the tensors in draw order, the two-way layers, the transposed convolution, the dot products and the four maps for each of the three prompts.
9. **[Implement]** Reproduce `segment_anything.memory_selection` and `segment_anything.memory_toy`: the slots and pointers of each listed frame and the attention weights with and without the rotation.
10. **[Implement]** Reproduce `segment_anything.hiera_stages` and `segment_anything.encoder_vs_decoder` from the per-block rule: the block table, its FLOPs per stage and the decoder's FLOPs by part.
11. **[Investigate]** Using the counts of the chapter, find the memory size $n_{\mathrm{mem}}$ at which the memory attention of a tracked frame costs as much as the trunk. What does that say about a longer bank?
12. **[Investigate]** Change the toy memory so that the object moves by two cells and then by three. Report the weight on the old cell and the read-back mask at the new cell with and without rotation, and say how far the object can move before the rotation makes the memory useless.

## Key terms

- **promptable segmentation**: the task of returning the mask of an object in an image from a prompt, a point, a box or a rough mask, with the same model for every object and no fixed list of classes; a prompt that leaves the extent open is answered with several masks. <!-- concept: promptable-segmentation -->
- **mask decoder**: the small network that turns the prompt tokens and the stored image embedding into mask logits, by letting tokens and image cells attend to each other and then taking, for each output token, a dot product with the upscaled cells. <!-- concept: mask-decoder -->
- **prompt encoder**: the part that turns clicks, box corners and a rough mask into vectors for the decoder: a random Fourier encoding of each coordinate plus a learned embedding of its type, and a convolutional reduction of a mask. <!-- concept: prompt-encoder -->
- **memory attention**: the attention of a video frame's features onto a bank of memories of earlier frames and object pointers, with position encoded by rotation, which marks the cells that resemble the tracked object before the mask is decoded. <!-- concept: memory-attention -->

## Answers

**Recall questions.**

1. The queries come from one sequence (the 7 tokens) and the keys and values from another (the 16 cells). The weight matrix is $7 \times 16$, each row summing to 1.
2. $\lceil 64/14 \rceil = 5$ per side, so 25 windows, and the padded grid has $70 \times 70 = 4{,}900$ rows.
3. $4/16 = 0.25$: the intersection is the small mask and the union is the large one.
4. On the offset $j - i$ alone: rotating $\mathbf{q}$ by $i\omega$ and $\mathbf{k}$ by $j\omega$ gives a score that depends on the relative rotation $(j - i)\omega$.
5. $8 \times 8$, and no output pixel receives more than one tap: with kernel size equal to stride the footprints tile the output.

**Check your understanding.**

1. The overlap is intersection over union, and the union includes the 6 extra pixels of the neighbour: $16/22 = 0.727$. A mask that is too big is penalized exactly as one that is too small. The decoder predicts the overlap of each of its masks, so a caller can see that the third candidate is not as good as the second.
2. A sine of a random projection of the coordinate is smooth only in the sense that it is continuous. Two points are given nearly the same code only if the frequencies are small compared with their distance, and $\sigma_\Omega$ sets the frequencies: by (4) the similarity falls as $\exp(-2\pi^2\sigma_\Omega^2\lVert\delta\rVert^2)$, so with a large $\sigma_\Omega$ nearby points are quickly told apart.
3. The token sequence is shorter by one (the self-attention, the token-to-image attention and the output tokens all see 7 tokens instead of 8). A prompt with no real point is still valid, and it is exactly the tracking prompt of the video path, which has one padding point for the label $-1$ and the appended one.
4. The attention compares where the cells are with where the tokens look, so the position belongs in the comparison. What is copied into the token is the content of the cell. Adding the position to the values would move a positional code into the tokens, which then would not be a description of the image.
5. The encoder-per-click cost is $(1{,}812.15 + 3.64)$ per click, 8 clicks cost 14,526.29 billion FLOPs, against 1,841.26 when the encoder runs once: 7.9 times more.
6. A bank that keeps every frame has 100 slabs, $409{,}600$ keys, at frame 100 (plus pointers), about 14 times the 28,736 of the bounded bank, so the cross-attention costs about 14 times more at that frame, and the memory held grows with the clip.
7. Without pointers the model would have the appearance and position of the object but not the decoder's summary of it, a vector that survives frames where the slab is stale. Without slabs the model would have only a few vectors with no grid, and could not tell where in the new frame the object is.

**Exercises.** 1. The similarity at 0.1 is $\exp(-2\pi^2 \cdot 4 \cdot 0.01) = 0.454$; at 0.25 it is $\exp(-2\pi^2 \cdot 4 \cdot 0.0625) = 0.0072$; it falls to one half where $2\pi^2 \cdot 4 \cdot \delta^2 = \ln 2$, $\delta = 0.0937$. 2. A box is two corner tokens and a click pair is two tokens, and the padding token is appended to each, so both have 3 prompt tokens and, with the 6 output tokens, 9 tokens in all (one click has 8). The weight matrix of one head is $9 \times 4{,}096$ in both cases. 3. $12 \cdot 2{,}304^2 \cdot 4 = 254{,}803{,}968$ bytes against $12 \cdot 2{,}304 \cdot 96 \cdot 4 = 10{,}616{,}832$, a ratio of 24; at $96 \times 96$ it is $12 \cdot 9{,}216^2 \cdot 4 = 4{,}076{,}863{,}488$ against $12 \cdot 9{,}216 \cdot 192 \cdot 4 = 84{,}934{,}656$, a ratio of 48. 4. $1 - 4{,}096/4{,}900 = 16.4$ percent. 5. At frame 3 the keys are $3 \cdot 4{,}096 + 3 \cdot 4 = 12{,}300$ and the score and average products are $4 \cdot 4{,}096 \cdot 12{,}300 \cdot 256 = 51.6$ GFLOPs per layer, 206.4 over four layers.
