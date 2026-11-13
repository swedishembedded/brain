<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Vision Transformers

*Can attention replace convolution for images?*

The transformer of Volume V takes a sequence of vectors and lets every one of them read every other. A picture is a grid of numbers, so the most direct way to feed one in would be to let every pixel be a token. Here is what that costs for a picture of ordinary size.

<!-- worked: vision_transformers.pixel_tokens -->
A 224 by 224 picture, the width D = 768 of the base block. Entries of one head's score matrix are T^2 for T tokens, stored as 32-bit numbers (4 bytes); the score and average products of one block cost 4 T^2 D FLOPs (Volume V, Chapter 9) for all heads together.

| tokens | T | score entries per head | bytes per head | FLOPs of the products, one block |
|---|---:|---:|---:|---:|
| one token per pixel | 50,176 | 2,517,630,976 | 10,070,523,904 | 7,734,162,358,272 |
| patches of 16 by 16 | 196 | 38,416 | 153,664 | 118,013,952 |

The ratio of the two token counts is 256, and of the pair counts 65,536.
<!-- /worked -->

The score matrix of one head would have two and a half billion entries, ten gigabytes at 32 bits, and that is for one of twelve heads in one of twelve layers. The products alone are 7.7 trillion FLOPs per block. Cutting the picture into patches of 16 by 16 pixels brings the token count down by a factor of 256 and the pair count by 65,536, to a size at which a block runs routinely. The rest of the chapter is about what that cut costs, what it needs from the model to make up for it, and the several ways, windows, pyramids and convolutional early stages, in which the count of tokens is kept small when the picture is large.

Six earlier ideas meet in the vision transformer, and the questions below collect them.

1. (Volume V, [Chapter 9](../v05-sequences/09-transformer-block.md), the transformer block.) A block of width $D$ reads $T$ tokens. How many entries does one head's score matrix have, and how many FLOPs do the score and weighted-average products of the whole block cost?
2. (Volume VII, [Chapter 2](02-convolution.md), convolution.) A convolution with a $2 \times 2$ kernel, stride 2 and 4 filters is applied to a map of shape $[3, 8, 8]$. What is the output shape, how many weights does it have and how many multiply-adds does it do?
3. (Volume V, [Chapter 8](../v05-sequences/08-positions.md), absolute positions.) Self-attention with no position information is given the rows of $\mathbf{X}$ in a different order. What happens to its output?
4. (Volume V, [Chapter 8](../v05-sequences/08-positions.md), multi-axis rotary embedding.) In a picture cut into patches that are numbered row by row, what offset separates a patch from the one below it, and what offset does multi-axis RoPE give?
5. (Volume VII, [Chapter 1](01-images.md), resampling.) A row of 4 values is resized to 6 under the half-pixel rule. At which source coordinate does output pixel 0 read?
6. (Volume VI, [Chapter 3](../v06-language-models/03-attention-variants.md), sliding-window attention.) In a causal window of $w$ tokens, how many keys does a query read, and how many (query, key) pairs does a sequence of $T \gg w$ tokens need?

## Cutting a picture into patches

The smallest picture that shows the idea has 8 by 8 pixels. To make every pixel identifiable, give the pixel in row $i$ and column $j$ the code $8i + j$, and cut the picture into squares of $P = 2$ pixels on a side. Each square is a **patch**; the patches form a grid of $n_p = 8/2 = 4$ rows and columns, and numbering them row by row gives token $t = 4a + b$ to the patch in row $a$ and column $b$.

<!-- worked: vision_transformers.patches -->
The picture of one channel whose pixel in row i, column j has the code 8 i + j:

```text
  0   1   2   3   4   5   6   7
  8   9  10  11  12  13  14  15
 16  17  18  19  20  21  22  23
 24  25  26  27  28  29  30  31
 32  33  34  35  36  37  38  39
 40  41  42  43  44  45  46  47
 48  49  50  51  52  53  54  55
 56  57  58  59  60  61  62  63
```

Cut into 2 by 2 patches, in row-major order of the patches. A patch is flattened row by row, so its four numbers are the pixels (2a, 2b), (2a, 2b + 1), (2a + 1, 2b), (2a + 1, 2b + 1).

| token t = 4a + b | patch (a, b) | pixel rows | pixel columns | x0 | x1 | x2 | x3 |
|---|---|---|---:|---:|---:|---:|---:|
| 0 | (0, 0) | 0-1 | 0-1 | 0 | 1 | 8 | 9 |
| 1 | (0, 1) | 0-1 | 2-3 | 2 | 3 | 10 | 11 |
| 2 | (0, 2) | 0-1 | 4-5 | 4 | 5 | 12 | 13 |
| 3 | (0, 3) | 0-1 | 6-7 | 6 | 7 | 14 | 15 |
| 4 | (1, 0) | 2-3 | 0-1 | 16 | 17 | 24 | 25 |
| 5 | (1, 1) | 2-3 | 2-3 | 18 | 19 | 26 | 27 |
| 6 | (1, 2) | 2-3 | 4-5 | 20 | 21 | 28 | 29 |
| 7 | (1, 3) | 2-3 | 6-7 | 22 | 23 | 30 | 31 |
| 8 | (2, 0) | 4-5 | 0-1 | 32 | 33 | 40 | 41 |
| 9 | (2, 1) | 4-5 | 2-3 | 34 | 35 | 42 | 43 |
| 10 | (2, 2) | 4-5 | 4-5 | 36 | 37 | 44 | 45 |
| 11 | (2, 3) | 4-5 | 6-7 | 38 | 39 | 46 | 47 |
| 12 | (3, 0) | 6-7 | 0-1 | 48 | 49 | 56 | 57 |
| 13 | (3, 1) | 6-7 | 2-3 | 50 | 51 | 58 | 59 |
| 14 | (3, 2) | 6-7 | 4-5 | 52 | 53 | 60 | 61 |
| 15 | (3, 3) | 6-7 | 6-7 | 54 | 55 | 62 | 63 |
<!-- /worked -->

A patch of $C$ channels and $P \times P$ pixels is flattened to a vector of $CP^2$ numbers. The order in which the pixels are listed is a convention, and the table uses channel first, then row, then column. The **patch embedding** multiplies that vector by one matrix and adds a bias,

$$
\mathbf{x}_t = \mathbf{W}_e\,\mathbf{v}_t + \mathbf{b}_e, \qquad \mathbf{v}_t \in \mathbb{R}^{C P^2},\ \mathbf{W}_e \in \mathbb{R}^{D \times C P^2}, \tag{1}
$$

so that every patch becomes a token of width $D$, the width of the transformer that follows. The same matrix is used for every patch. A picture of side $S$ becomes $T = (S/P)^2$ tokens, and the embedding has $D\,CP^2 + D$ parameters whatever the picture size. As an algorithm, with the flatten order of the table:

```text
function patchify(x: array<real>[C, S, S], P: int) -> array<real>[(S // P) * (S // P), C * P * P]:
    require S mod P == 0
    n = S // P
    out = zeros(real, [n * n, C * P * P])
    for a in 0 .. n:
        for b in 0 .. n:
            for c in 0 .. C:
                for i in 0 .. P:
                    for j in 0 .. P:
                        out[a * n + b, (c * P + i) * P + j] = x[c, a * P + i, b * P + j]
    return out
```

The tokens are `matmul(patchify(x, P), transpose(We)) + be`, with the bias added to every row.

![An 8 by 8 picture of pixel codes, cut into 2 by 2 patches with patch (1, 3) highlighted; the patch is flattened row by row to the four numbers 22, 23, 30, 31, multiplied by W and given a bias to become token 7, which sits in a strip of 16 tokens numbered row by row.](img/patchify-embedding.svg)

The choice of $P$ is a trade between detail and length. A patch is embedded by one linear map, so everything inside it that is finer than the map can express is lost before attention starts: a patch of 16 by 16 pixels and 3 channels is 768 numbers, and they must all fit into one token of the model's width. A smaller patch keeps more and costs quadratically more attention, which a later section counts.

## The embedding is a convolution

The embedding of (1) looks like a different operation from the convolution of Chapter 2, and it is the same one. A convolution with kernel $P$ and stride $P$ reads non-overlapping windows of $P \times P$ pixels, which are exactly the patches, and computes at each of them the dot product of the window with each filter. Reshape the rows of $\mathbf{W}_e$ into filters $K[o, c, i, j] = \mathbf{W}_e[o, (cP + i)P + j]$ and the two agree:

$$
x_t[o] = b_e[o] + \sum_{c=0}^{C-1}\sum_{i=0}^{P-1}\sum_{j=0}^{P-1} K[o, c, i, j]\; I[c,\ Pa + i,\ Pb + j]. \tag{2}
$$

Here is a numerical check on a picture of 3 channels, with the matrix applied to the flattened patches and the convolution `corr2d` of Chapter 2 applied to the whole picture, and what happens if the flattening order is not the one the convolution implies.

<!-- worked: vision_transformers.embedding_equivalence -->
seed 0x3018484de019d48a (3465599412571067530). Draws, in this order: the 3 * 8 * 8 = 192 pixels x[c][i][j] (channel, row, column), each `round(normal(), 1)`; then the matrix W of 4 rows (output o) and 12 columns, row by row, each `round(0.5 * normal(), 1)`, whose column index is (c * 2 + i) * 2 + j for the pixel (i, j) of channel c in the patch; then the bias b, 4 numbers, each `round(0.5 * normal(), 1)`.

| W | col 0 | col 1 | col 2 | col 3 | col 4 | col 5 | col 6 | col 7 | col 8 | col 9 | col 10 | col 11 | b |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| o = 0 | -0.2 | -0.1 | -0.2 | -0.4 | -0.3 | -0.5 | 0.5 | -0.1 | -0.2 | -1.0 | -0.4 | -0.3 | 0.4 |
| o = 1 | 1.1 | 0.4 | -0.4 | -0.6 | 0.0 | 0.5 | 0.2 | -0.1 | -0.5 | -0.3 | 0.0 | -0.6 | 0.6 |
| o = 2 | -0.8 | 0.5 | -0.2 | -1.0 | 0.1 | 0.0 | 0.2 | 0.0 | 0.5 | 1.6 | -0.5 | 0.4 | -0.1 |
| o = 3 | 0.6 | -0.5 | 0.0 | 0.1 | 0.9 | -0.1 | -0.1 | -1.2 | 0.7 | -0.7 | -0.6 | 0.3 | 0.1 |

The 16 tokens (patch t = 4a + b is the patch in row a, column b), computed as a matrix product on the flattened patch, each sum left to right over the 12 numbers, then plus the bias:

| t | token[0] | token[1] | token[2] | token[3] |
|---|---:|---:|---:|---:|
| 0 | 1.1600 | 2.1400 | -0.2600 | 2.3100 |
| 1 | 0.7000 | 1.0000 | 0.4200 | -1.8200 |
| 2 | -0.7300 | 3.3100 | 1.7000 | -3.9700 |
| 3 | 2.2400 | 1.3100 | -1.1900 | -3.6200 |
| 4 | -0.6900 | 3.0900 | -0.8300 | -1.2000 |
| 5 | 1.6800 | -0.4400 | -3.0700 | -1.7000 |
| 6 | 0.1500 | -1.5600 | -0.1100 | 1.5700 |
| 7 | 2.0700 | -0.7000 | -1.9000 | 1.2600 |
| 8 | 1.2200 | 3.0900 | -2.8500 | 0.9100 |
| 9 | 0.4300 | 0.7200 | -2.6600 | -2.3900 |
| 10 | 0.6600 | -0.2600 | -0.0700 | 0.3500 |
| 11 | 0.9600 | -0.1400 | 0.8100 | -2.9600 |
| 12 | 1.6700 | 1.2600 | 1.5400 | -1.0800 |
| 13 | 1.3700 | -0.6600 | -0.2600 | -1.9000 |
| 14 | 1.4000 | 0.7600 | 0.0600 | -0.9100 |
| 15 | 0.1900 | 0.6100 | -0.5700 | 2.8800 |

The same 64 numbers as a convolution of the [3, 8, 8] picture with the 4 filters W[o] reshaped to [3, 2, 2], stride 2, no padding (`corr2d` of Volume VII, Chapter 2), plus the bias: the largest absolute difference is below 1e-12. With the pixels flattened in the other order (row, column, channel) and the same W the largest absolute difference from the tokens above is 5.290: both orders are matrix products, and only one of them is this convolution.
<!-- /worked -->

The practical lesson is in the last sentence of the block. A patch embedding that is implemented as a matrix product on flattened patches and one that is implemented as a convolution are interchangeable only if the flatten order and the filter layout are the same, and a checkpoint trained with one carries a weight whose columns are in that order. Brain has both implementations, side by side. The SAM tower begins with a convolution whose kernel equals its stride (the header of `crates/sam1/src/lib.rs` writes it as `Conv2d(3, C, k=patch, s=patch)`), and the CLIP tower's weights are stored in the convolution layout `[width, 3, patch, patch]` (`patch_embed.weight` in `crates/clip/src/config.rs`, where the table of stems also records that the CLIP embedding has no bias and the SigLIP one has; a bias is part of a checkpoint, not of the idea). The Qwen3-VL tower packs patches on the host and multiplies (`pack_patches` in `crates/qwen3vl/src/preprocess.rs`), in the order channel, temporal slice, row, column, which is the order a three-dimensional convolution over a pair of frames would imply; a test of that file, `pack_single_patch_flattens_channel_temporal_ph_pw`, pins the order on a $2 \times 2$ picture, and another, `pack_two_channels_are_channel_major`, pins the channel position. The vector it embeds has $3 \cdot 2 \cdot 16 \cdot 16 = 1536$ numbers (`VisionConfig::patch_vec_dim`).

The identity has one more use, further on. Any operation that takes non-overlapping $k \times k$ groups of positions, concatenates their features and applies one matrix is a convolution with kernel and stride $k$.

## A transformer that reads every patch

With the tokens made, the rest is Volume V. The sequence of patch embeddings, with position information added (the next section), goes through a stack of the block of Chapter 9 of that volume with one difference: no causal mask. A patch has no future; every patch reads every other. The result is a **vision transformer**: a patch embedding, positions, a stack of unmasked transformer blocks, and a readout that turns the final token rows into whatever the task wants. The block is unchanged, so its accounting is the one the previous volume derived: the four projections cost $8TD^2$ FLOPs, the feed-forward network $4TDF$ (with the plain two-matrix network), the score and average products $4T^2D$ and the softmax $5HT^2$. The parameter count of a block does not depend on $T$ at all. Two details do differ between the book's block and the ones in the ViT towers. The book's block uses RMS normalization, as modern language models do; the original vision transformer, the CLIP tower and brain's FastViTHD attention block use layer normalization with a mean subtraction and a bias. The feed-forward network is a plain GELU network of hidden width $4D$ in the original and a gated one in some later towers (the EVA02 tower of `crates/clip` uses a gated network of hidden width 2,730 for $D = 1024$, per its configuration).

The readout is the new design decision. A language model reads off the last position, which is the only one that has seen every token. An encoder has two natural choices, and brain's CLIP crate holds both (`VisionStem` in `crates/clip/src/config.rs`). One is a **class token**: an extra learned vector placed before the patches, with no patch under it, that every layer lets read all the patches, and whose final row is the picture's summary. The CLIP stem prepends one, and its table of learned positions has $1 + n_p^2$ rows, 257 for a $16 \times 16$ grid. The other is to average the final rows of all the patches, which needs no extra token, and the SigLIP stem of the same crate has none. Which is better is an empirical question. The small experiment near the end of this chapter trains both ways.

## Where a patch is

Attention treats its input as a set, and Chapter 8 of Volume V showed what that means: permute the rows and the output rows come out permuted in the same way, so the model cannot tell a picture from a shuffle of its own patches. Something must say where each patch is, and a grid has two axes where a sentence has one.

**Learned positions.** The simplest answer is a table with one learned row for each patch position, added to the patch embedding: $\mathbf{x}_t \leftarrow \mathbf{x}_t + \mathbf{E}_t$. It is what the original vision transformer does, and CLIP, SigLIP and the SAM tower in brain. The table is learned for a particular grid and is the subject of the next section.

**Sinusoids on two axes.** The fixed encoding of Chapter 8 of Volume V extends to a grid by giving half the channels to the row and half to the column. Take $D = 8$ and look at how alike the encoding of the patch in row 0, column 0 is to the encoding of each other patch of a $4 \times 4$ grid, measured by the dot product:

<!-- worked: vision_transformers.grid_offsets -->
Width D = 8, base 10000 (`positions.sinusoid`, Volume V, Chapter 8). The dot product of the encoding of the patch in row 0, column 0 with the encoding of every patch of a 4 by 4 grid.

Two axes: the first four channels are the sinusoid of the row (width 4), the last four of the column.

|  | col 0 | col 1 | col 2 | col 3 |
|---|---:|---:|---:|---:|
| row 0 | 4.000 | 3.540 | 2.584 | 2.010 |
| row 1 | 3.540 | 3.081 | 2.124 | 1.550 |
| row 2 | 2.584 | 2.124 | 1.167 | 0.593 |
| row 3 | 2.010 | 1.550 | 0.593 | 0.019 |

One axis: the patches numbered 4 r + c in row-major order and encoded as positions 0 .. 15 of a width-8 sinusoid.

|  | col 0 | col 1 | col 2 | col 3 |
|---|---:|---:|---:|---:|
| row 0 | 4.000 | 3.535 | 2.564 | 1.965 |
| row 1 | 2.267 | 3.160 | 3.784 | 3.516 |
| row 2 | 2.548 | 1.706 | 1.696 | 2.452 |
| row 3 | 3.199 | 3.166 | 2.297 | 1.300 |
<!-- /worked -->

On two axes the picture is geometric: the patch to the right (0, 1) and the patch below (1, 0) are exactly as close as each other, and the score falls as the offset grows on either axis. On one axis, with the patches numbered 0 to 15 and encoded as a line, the patch directly below (number 4) scores 2.267, less than the patch two to the right (number 2, 2.564), and the score does not fall steadily at all. The line has no way to say "below".

**Rotary embeddings on two axes.** The rotation of Chapter 8 of Volume V extends the same way. A head of width $d$ has $d/2$ rotation pairs; give the first $d/4$ to the row and the last $d/4$ to the column, and let pair $m$ of an axis turn by the position on that axis times its own rate. In the half-split pairing (pair $m$ is channels $m$ and $m + d/2$) and with the rates $\omega_j = \theta^{-4j/d}$ for $j = 0, \dots, d/4 - 1$, restarting for each axis:

$$
\text{pair } m < d/4:\ \text{angle } r\,\omega_m, \qquad \text{pair } m \ge d/4:\ \text{angle } c\,\omega_{m - d/4}, \tag{3}
$$

for the patch in row $r$ and column $c$. This is what the Qwen3-VL tower of brain computes (`vision_rope_tables` in `crates/qwen3vl/src/vision.rs`: the angle vector of a token is the row times the inverse frequencies followed by the column times the same inverse frequencies, $\theta = 10000$, applied half-split). As in one dimension, the dot product of a rotated query and a rotated key depends only on the offset, now an offset in rows and columns:

<!-- worked: vision_transformers.rope_2d_scores -->
seed 0xe1e8f397711fe5df (16278528684894184927). Draws: the 8 entries of q, then the 8 of k, each `round(normal(), 1)`; q = ['1.3', '-0.5', '-2.5', '1.7', '-2.5', '-0.3', '-0.4', '0.3'], k = ['-0.3', '0.3', '-0.5', '2.2', '1.9', '0.5', '0.4', '-0.1']. Head width 8, base 10000, half-split pairs (m, m + 4); pairs 0 and 1 turn by the row times the rates 1 and 0.01, pairs 2 and 3 by the column times the same two rates. The score is the dot product of the rotated query at (row, column) with the rotated key.

| query at | key at | offset (rows, columns) | score |
|---|---|---|---:|
| (0, 0) | (1, 2) | (1, 2) | -0.159455 |
| (3, 1) | (4, 3) | (1, 2) | -0.159455 |
| (5, 7) | (6, 9) | (1, 2) | -0.159455 |
| (1, 2) | (0, 0) | (-1, -2) | 0.516493 |
| (0, 0) | (2, 1) | (2, 1) | 5.595072 |
| (0, 0) | (0, 3) | (0, 3) | -2.616521 |
| (0, 0) | (3, 0) | (3, 0) | 9.350769 |
<!-- /worked -->

The first three rows have the same offset $(1, 2)$ at three different places and the same score to six digits. The offset $(2, 1)$ of the fifth row gives another score, as it should (a row is not a column), and so does the reversed offset $(-1, -2)$ of the fourth row. Rotary embeddings add nothing to the tokens and have no table, so they apply at any grid size, and they have a second advantage the next section needs: a model trained at one grid has nothing to resample. The EVA02 tower in `crates/clip` records the price, which is a stretch: it was pretrained on a grid of `pt_seq_len` = 16 patches on a side and runs at 24 on a side, so its configuration scales the positions by 16/24 (the comment in `EvaVisionConfig` gives $t = \mathrm{arange}(g)/g \cdot \mathrm{pt}$). The same idea as the interpolation of Volume V, Chapter 8.

**Positions from a convolution.** A fourth way avoids a table and a rotation altogether. A depthwise convolution with zero padding sees the border of the picture and, in a deeper layer, where it is, because a position near the edge sees zeros that an interior position does not; a layer that adds its output to the tokens therefore tells each token something about where it is. Brain's FastViTHD does exactly this before each of its attention stages, with a depthwise $7 \times 7$ convolution that the code calls RepCPE (`repcpe` in `crates/fastvlm/src/encoder.rs`), and its attention blocks carry no position information of their own.

## When the grid changes

A table of learned positions has a fixed number of rows. A model trained on $224 \times 224$ pictures with 16-pixel patches has a $14 \times 14$ grid of positions, and is asked to read a $384 \times 384$ picture with a $24 \times 24$ grid. The original vision transformer fine-tunes at the higher resolution and resamples the table, as a picture: the learned positions are laid out on their grid and interpolated in two dimensions to the new grid. This is image resampling in the sense of Chapter 1, with the table as the image and each channel as a plane.

The conventions are the whole difficulty, and the three towers of brain that do it use three. Here is a toy table, of one channel on a $4 \times 4$ grid, resampled to $6 \times 6$ the way brain's CLIP tower does it: cubic convolution with $a = -0.75$ and source coordinate

$$
\xi_o = (o + \tfrac12)\,\frac{n_{\mathrm{in}}}{n_{\mathrm{out}}} - \tfrac12, \tag{4}
$$

the four taps $\lfloor\xi_o\rfloor - 1, \dots, \lfloor\xi_o\rfloor + 2$ with indices clamped to the table, and no renormalization (the kernel header `crates/kernels/wgsl/resize_bicubic.wgsl` is explicit about each of the three choices, and the CLIP model calls it with align corners off in `crates/clip/src/model.rs`). The kernel is the cubic convolution of Chapter 1, a function of the distance $s \ge 0$: $k_a(s) = (a + 2)s^3 - (a + 3)s^2 + 1$ for $s \le 1$, $k_a(s) = as^3 - 5as^2 + 8as - 4a$ for $1 < s < 2$ and 0 beyond, and the pseudocode passes it the distance `abs(t - k)`. One axis of the resample is:

```text
function cubic_taps(n_in: int, n_out: int, a: real) -> list<list<(int, real)>>:
    result = []
    for o in 0 .. n_out:
        xi = (o + 0.5) * n_in / n_out - 0.5
        f = floor(xi)
        t = xi - f
        taps = []
        for k in -1 ..= 2:
            append (min(max(f + k, 0), n_in - 1), k_a(abs(t - k))) to taps
        append taps to result
    return result
```

The table is resampled along each row with these taps and then down each column.

<!-- worked: vision_transformers.position_resample -->
seed 0x1cd5e1b003ae31a9 (2077814949193593257). A learned table for a 4 by 4 grid with one channel: 16 values, row by row, each `round(normal(), 1)`.

| table | col 0 | col 1 | col 2 | col 3 |
|---|---:|---:|---:|---:|
| row 0 | 1.4 | 0.3 | 1.1 | -1.1 |
| row 1 | 0.5 | -0.9 | 1.1 | -2.7 |
| row 2 | -0.7 | -0.8 | 0.5 | 0.4 |
| row 3 | -2.1 | 2.0 | 0.0 | -0.1 |

Resampled to 6 by 6 as the CLIP tower does (cubic convolution with a = -0.75, source coordinate (o + 0.5) * 4 / 6 - 0.5, the four taps floor(s) - 1 .. floor(s) + 2 with indices clamped to the table, no renormalization), along each row first and then down each column. The taps of one axis:

| output o | source s | taps | w(-1) | w(0) | w(+1) | w(+2) | sum |
|---:|---:|---|---:|---:|---:|---:|---:|
| 0 | -0.1667 | -2 .. 1 | -0.0174 | 0.1609 | 0.9433 | -0.0868 | 1.0000 |
| 1 | 0.5000 | -1 .. 2 | -0.0938 | 0.5938 | 0.5938 | -0.0938 | 1.0000 |
| 2 | 1.1667 | 0 .. 3 | -0.0868 | 0.9433 | 0.1609 | -0.0174 | 1.0000 |
| 3 | 1.8333 | 0 .. 3 | -0.0174 | 0.1609 | 0.9433 | -0.0868 | 1.0000 |
| 4 | 2.5000 | 1 .. 4 | -0.0938 | 0.5938 | 0.5938 | -0.0938 | 1.0000 |
| 5 | 3.1667 | 2 .. 5 | -0.0868 | 0.9433 | 0.1609 | -0.0174 | 1.0000 |

The resampled table:

|  | col 0 | col 1 | col 2 | col 3 | col 4 | col 5 |
|---|---:|---:|---:|---:|---:|---:|
| row 0 | 1.571 | 0.876 | 0.447 | 1.160 | 0.135 | -1.140 |
| row 1 | 1.182 | 0.239 | -0.160 | 1.213 | -0.380 | -2.481 |
| row 2 | 0.388 | -0.575 | -0.797 | 1.000 | -0.488 | -2.681 |
| row 3 | -0.365 | -0.910 | -0.879 | 0.430 | 0.460 | -0.086 |
| row 4 | -1.697 | -0.413 | 0.730 | 0.269 | 0.278 | 0.462 |
| row 5 | -2.609 | 0.225 | 2.304 | 0.371 | -0.308 | -0.152 |

Resampled to its own size the table is reproduced exactly: the largest difference is below 1e-12. The same table by the antialiased resampler of the SAM tower (a = -0.5, window clipped at the border and renormalized) differs from the table above by at most 0.237, and by bilinear interpolation with the corners aligned (source coordinate o * 3 / 5, the convention of the Qwen3-VL tower) by at most 0.704.
<!-- /worked -->

The SAM tower's convention is the antialiased bicubic of Chapter 1 with $a = -0.5$ (`resample_pos_embed` in `crates/sam1/src/resample.rs` calls the shared `imaging::host::resize_aa_planar`), and the Qwen3-VL tower interpolates bilinearly with the corners aligned, source coordinate $o\,(n_{\mathrm{in}} - 1)/(n_{\mathrm{out}} - 1)$ (`pos_embed_bilinear` in `crates/qwen3vl/src/vision.rs`). The block shows how much they disagree on a table of values of order one: up to 0.237 between the CLIP and SAM rules, 0.704 between the CLIP rule and the corner-aligned bilinear rule. A position table is resampled once, at load, in the SAM crate, and as a step of the graph, on every forward pass, in the CLIP crate; either way the choice is part of the model, as the resizer of Chapter 1 is part of its preprocessing. A tower that resamples with the wrong rule produces features that are plausible and wrong.

## What a token costs

Now the count that decides the design. For the base block of the original ViT ($D = 768$, $H = 12$, hidden width 3,072) here are the FLOPs of the projections and feed-forward network, and of attention, at the sizes the towers above use.

<!-- worked: vision_transformers.token_costs -->
One block of width D = 768, H = 12 heads, a GELU feed-forward network of hidden width 3072 (the base size of the original ViT). Every FLOP count is the rule of Volume V, Chapter 9: 8 T D^2 for the four projections, 2 * 2 T D F for the feed-forward network, 4 T^2 D for the score and average products and 5 H T^2 for the softmax; the other terms of that chapter's count (two norms 2 T (4 D + 3), two adds 2 T D and the GELU 5 T F, in all T (10 D + 6 + 5 F) FLOPs) are under 1 percent, are left out of the first two FLOP columns and are counted in the block total that the attention share divides by. FLOPs are in billions. The memory is the 32-bit score matrices of all H heads, in MiB, if they are materialized. A class token would add one to T.

| side (px) | patch (px) | grid | tokens T | scores (MiB) | projections and feed-forward (GFLOPs) | attention: products and softmax (GFLOPs) | attention share of the block (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 224 | 16 | 14 x 14 | 196 | 1.8 | 2.775 | 0.120 | 4.1 |
| 384 | 16 | 24 x 24 | 576 | 15.2 | 8.154 | 1.039 | 11.3 |
| 512 | 16 | 32 x 32 | 1,024 | 48.0 | 14.496 | 3.284 | 18.4 |
| 1024 | 16 | 64 x 64 | 4,096 | 768.0 | 57.982 | 52.546 | 47.5 |
| 224 | 8 | 28 x 28 | 784 | 28.1 | 11.098 | 1.925 | 14.8 |
| 224 | 4 | 56 x 56 | 3,136 | 450.2 | 44.393 | 30.802 | 40.9 |
<!-- /worked -->

Doubling the side quadruples the tokens, which quadruples the projections and the feed-forward network and multiplies attention by 16: from 512 to 1,024 pixels the first group goes from 14.496 to 57.982 billion FLOPs and attention from 3.284 to 52.546 billion. The products of attention overtake the projections and the feed-forward network where $4T^2D = 8TD^2 + 4TDF$, at $T = 2D + F = 4{,}608$ tokens for this block, a grid of about $68 \times 68$ patches, a picture of about 1,086 pixels on a side. At 224 pixels, attention is 4.1 percent of the block; at 1,024 it is 47.5 percent. The memory is the other side of it: the score matrices of all twelve heads at 4,096 tokens are 805,306,368 bytes (768 MiB) if they are materialized, per layer, and Chapter 13 of Volume V explained how to avoid materializing them. Avoiding the memory does not avoid the FLOPs.

Brain's Qwen3-VL tower attends over the whole image at once, and the cost shows in its code. `ATTN_SLAB_BUDGET` in `crates/qwen3vl/src/encoder.rs` caps one score slab at 256 MiB and dispatches the queries in chunks. The comment says why the result is unchanged: the chunks split the query rows only, never the keys, so each row still takes its softmax over the whole image, in the same order. The crate's other cost lever is the size of the grid, which the dynamic-resolution rule of Chapter 9 will control.

## Windows

If every token reads every token, the cost is $T^2$. If every token reads only the tokens within a window of $M \times M$ patches, and the windows do not overlap, the picture is cut into $T/M^2$ windows, each costing $(M^2)^2$ pairs, so the total is $T M^2$ and linear in $T$. This is **windowed attention**, the two-dimensional sibling of the sliding window of Volume VI: there the window slides with the query, here the windows tile the grid.

<!-- worked: vision_transformers.windows_cost -->
Pairs of (query, key) per head for global attention (T^2) and for non-overlapping windows of m by m (the grid divides evenly: T / m^2 windows of m^2 tokens, each with m^4 pairs, so T m^2 in all; the last column recomputes it as T m^2).

| grid | window | tokens T | global pairs | window pairs | global / window | T m^2 |
|---|---:|---:|---:|---:|---:|---:|
| 8 x 8 | 4 x 4 | 64 | 4,096 | 1,024 | 4.0 | 1,024 |
| 14 x 14 | 7 x 7 | 196 | 38,416 | 9,604 | 4.0 | 9,604 |
| 28 x 28 | 7 x 7 | 784 | 614,656 | 38,416 | 16.0 | 38,416 |
| 56 x 56 | 7 x 7 | 3,136 | 9,834,496 | 153,664 | 64.0 | 153,664 |
<!-- /worked -->

The ratio of global to windowed pairs is exactly $T/M^2$, the number of windows: 4 for a $14 \times 14$ grid of $7 \times 7$ windows and 64 for a $56 \times 56$ grid. The price is that a token cannot see beyond its window, in a layer. A token near a window's edge has no access to its neighbour across the line. Two remedies are in use. Brain's SAM tower is a case of the first, and the shared builder carries the planner of the second.

The first is a few global blocks among the windowed ones, which is what the SAM tower of `crates/sam1` does. Its configuration, `SamViTConfig::deepseek_ocr`, has 12 blocks of width 768 on a $64 \times 64$ grid with windows of $14 \times 14$ and blocks 2, 5, 8 and 11 global. $64$ is not a multiple of 14, so the grid is padded with zeros to $70 \times 70$ and cut into 25 windows of 196 tokens (`WindowPlan::padded` in `crates/model/src/vit.rs`).

<!-- worked: vision_transformers.sam_windows -->
The tower of `SamViTConfig::deepseek_ocr`: width 768, 12 blocks, a 64 by 64 grid, windows of 14 by 14 on the grid padded with zeros to a multiple of 14, blocks 2, 5, 8 and 11 global. Products FLOPs are 4 * pairs * 768 per block.

| quantity | shape | value |
|---|---|---:|
| tokens of the grid | 64 x 64 | 4,096 |
| grid padded to a multiple of the window | 70 x 70 | 4,900 |
| windows (each 14 x 14 = 196 tokens) | 5 x 5 | 25 |
| pairs of a global block |  | 16,777,216 |
| pairs of a windowed block, padded windows |  | 960,400 |
| product FLOPs, all 12 blocks global |  | 618,475,290,624 |
| product FLOPs, 8 windowed and 4 global blocks |  | 229,761,220,608 |
| ratio |  | 2.69 |
| share of the 4 global blocks in the mixed total (%) |  | 89.7 |
<!-- /worked -->

Four global blocks out of twelve are 89.7 percent of the attention products of the whole tower: the padded windows cost 960,400 pairs against 16,777,216 for a global block. If the tower were all global it would cost 2.69 times what it does.

## Shifting the windows

The second remedy keeps all blocks windowed and moves the windows between layers. Swin (Liu et al., 2021) alternates two partitions: a regular one, and one in which the windows are displaced by $u = M/2$ patches along both axes. A token at the edge of a window in one layer is in the interior of a window in the next. The cost is the same $TM^2$; what changes is the reach. The shifted partition has a complication, which is what happens at the picture's border: displacing the grid leaves partial windows at the edges. Swin avoids ragged windows by rolling the grid cyclically, so that the tokens pushed off one edge reappear on the opposite one and every window is again $M \times M$; the rolled windows then contain tokens that were not neighbours in the picture (those that wrapped around), and attention among them must be forbidden by a mask.

Deriving that mask on a grid small enough to print makes it concrete. Take a $4 \times 4$ grid of tokens, windows of $M = 2$ and a shift of $u = 1$. Roll the grid by $-u$ along both axes. Then label each position of the rolled grid by where along each axis it came from: the index $i$ is in region 0 if $i < n_p - M$, in region 1 if $n_p - M \le i < n_p - u$ and in region 2 otherwise, and the label is $3 \times (\text{row region}) + (\text{column region})$. Two positions of a window may attend to each other if and only if their labels are equal:

$$
\mathrm{mask}[x, y] = \begin{cases} 0 & \text{if } \mathrm{label}(x) = \mathrm{label}(y), \\ -\infty & \text{otherwise,} \end{cases} \tag{5}
$$

added to the scores before the softmax.

```text
function shift_labels(n: int, M: int, u: int) -> array<int>[n, n]:
    require 0 <= u and u < M and n mod M == 0
    label = zeros(int, [n, n])
    for i in 0 .. n:
        for j in 0 .. n:
            ri = 0 if i < n - M else (1 if i < n - u else 2)
            rj = 0 if j < n - M else (1 if j < n - u else 2)
            label[i, j] = 3 * ri + rj
    return label
```

<!-- worked: vision_transformers.shifted_window_mask -->
A 4 by 4 grid of tokens numbered 0 .. 15 row by row, windows of m = 2 by 2, shift s = 1. Swin rolls the map by -s along both axes (the token at row i, column j of the rolled map is the token at row (i + s) mod 4, column (j + s) mod 4 of the original), cuts it into windows, and labels each position of the rolled map by the region it came from: along each axis the index i is in region 0 if i < n - m, 1 if n - m <= i < n - s and 2 otherwise, and the label is 3 * (row region) + (column region).

Token ids of the rolled map:

```text
 5  6  7  4
 9 10 11  8
13 14 15 12
 1  2  3  0
```

Region labels of the rolled map:

```text
 0  0  1  2
 0  0  1  2
 3  3  4  5
 6  6  7  8
```

| window (row, column) | token ids | region labels | allowed pairs of 16 |
|---|---|---|---|
| (0, 0) | 5 6 9 10 | 0 0 0 0 | 16 |
| (0, 1) | 7 4 11 8 | 1 2 1 2 | 8 |
| (1, 0) | 13 14 1 2 | 3 3 6 6 | 8 |
| (1, 1) | 15 12 3 0 | 4 5 7 8 | 4 |

The mask of the last window (entry [u][v] is 1 when position u may read position v, i.e. their labels are equal):

```text
1 0 0 0
0 1 0 0
0 0 1 0
0 0 0 1
```

The tokens of one region are exactly the groups of the ragged partition that cuts each axis at [0, s), then every m, i.e. at rows and columns {0}, {1, 2}, {3}: its 9 groups have sizes [4, 2, 2, 2, 2, 1, 1, 1, 1]. Single-head attention (width 4, scale 1/2) on random q, k, v (seed 0xed71388a0ba3de1c (17109518624909155868); draws: the 64 entries of q, then of k, then of v, token by token, each `round(normal(), 1)`), computed once by the rolled windows with the mask and once by the ragged groups with no mask: the largest absolute difference of the 16 outputs is below 1e-12. Against the unshifted 2 by 2 windows the outputs differ by at most 2.391, and against global attention by at most 2.962.
<!-- /worked -->

The two partitions of an $8 \times 8$ grid and the mask of the last rolled window, for $M = 4$ and $u = 2$:

![Three panels. Left: an 8 by 8 grid cut into four regular windows of 4 by 4. Middle: the same grid after a shift of 2, cut into nine ragged groups of widths 2, 4 and 2 on each axis. Right: the 16 by 16 attention mask of the last rolled window, dark where two positions share a region, which has four blocks of 16 allowed pairs among 256.](img/window-partitions.svg)

Look at what the mask does: the tokens of one region are exactly the cells of the ragged partition, the one that cuts each axis at the shift, then at every $M$ patches from there on. The mask is a way to get that partition out of the roll. Brain makes use of the equivalence by not rolling at all. `WindowPlan::shifted` in `crates/model/src/vit.rs` builds the ragged windows directly, as a row permutation and a list of spans of different lengths, so that the same span-wise attention that runs the regular partition runs the shifted one, with no masked softmax; its comment says that Swin's masked partition "is EXACTLY the ragged partition". The worked block above checks it: the same random queries, keys and values attended once through the rolled windows with the mask (5) and once through the ragged groups, and the outputs agree to below $10^{-12}$. The crate's tests check that the plan is a permutation that covers every token once (`window_plan_partitions_every_row_once` in `crates/model/tests/it/vit_windowed.rs`, with shifts included) and that the block over ragged spans equals the unchunked block over the same spans (`shifted_window_cached_block_matches_unchunked`); the equivalence with roll and mask is the claim of the comments, and the block above is an independent computation of it.

The saving in pairs is not only a matter of tidiness. Counting pairs on an $8 \times 8$ grid for $M = 4$, $u = 2$:

<!-- worked: vision_transformers.window_pair_counts -->
Pairs of (query, key) of one head on an 8 by 8 grid of tokens, windows of 4 by 4, shift 2 where shifted.

| variant | pairs |
|---|---:|
| global attention | 4,096 |
| regular windows of 4 by 4 | 1,024 |
| rolled windows with a mask (every query scans its whole 4 by 4 window) | 1,024 |
| ragged windows (sizes 2, 4, 2 on each axis) | 576 |
<!-- /worked -->

A rolled window with a mask computes all $M^4$ pairs of its window and then throws most of them away; the ragged windows compute only the pairs that survive the mask, 576 against 1,024. The ragged windows are the product of the cuts of the two axes, so their pair count is the square of the sum of the squared cut lengths of one axis, $\bigl(\sum_k \ell_k^2\bigr)^2 = (2^2 + 4^2 + 2^2)^2 = 576$.

The windows have a cost in engineering, which is the one story of this part of brain's history. Commit `80178df9a` (July 2026) added `WindowPlan`, windowed spans and the pooling of queries to the shared ViT builder, so that Swin-style, Hiera-style and SAM-style towers could compose from one attention instead of each porting its own. Its message records two alignment facts that the ragged windows made load-bearing: a device binds a buffer at an offset that must be a multiple of 256 bytes, which is 64 floats, so a slab of $H \ell^2$ scores for a span of length $\ell$ is not bindable unless $H\ell^2$ is a multiple of 64. The message adds that a uniform case was already broken by it: 16 heads on a window of 729 tokens (the $27 \times 27$ patch grid of another vision encoder in the crate set) give $16 \cdot 729^2 = 8{,}503{,}056$ floats, which is $64 \cdot 132{,}860 + 16$. The fix pads every slab to a multiple of 64 floats (`probs_offsets` in `crates/model/src/vit.rs`). The lesson is the one the previous chapters keep finding: a general partition exposed an assumption that a regular one had been hiding.

## A pyramid of tokens

Windows keep attention cheap at a fixed grid. They do not shrink it. A convolutional network does shrink it: Chapter 3 followed a ResNet through strides 4, 8, 16 and 32, the feature pyramid that detectors read, with the width doubling each time the map halves. The same shape works for transformers. Swin's version is **patch merging**: concatenate the features of each $2 \times 2$ group of neighbouring tokens, which is $4C$ numbers, and map them with a linear layer to $2C$. The grid has a quarter of the tokens and twice the width.

By the identity of the second section, patch merging is a convolution of kernel 2 and stride 2, in other words a patch embedding applied to a feature map instead of to a picture:

<!-- worked: vision_transformers.merging_equivalence -->
seed 0x72a196195c3c5b37 (8260048227240008503). A feature map of 4 channels on a 4 by 4 grid: the 64 values x[c][i][j] (channel, row, column), each `round(normal(), 1)`, then a matrix of 8 rows and 16 columns, row by row, each `round(0.5 * normal(), 1)`, no bias. Merging concatenates the 4 channels of each 2 by 2 group of positions in the order (channel, row, column) into 16 numbers and multiplies by the matrix.

Output: 4 tokens of 8 numbers, the map [8, 2, 2]. The same operation as a convolution with 8 filters of shape [4, 2, 2], stride 2: the largest absolute difference over the 32 numbers is below 1e-12.
<!-- /worked -->

A transformer that does this between stages is a **hierarchical vision transformer**: a patch embedding of small patches at the start, then stages that each run blocks at one grid and one width, with a merge between stages. Here is Swin-T on a $224 \times 224$ picture, with the cost of its attention products stage by stage, windowed against global.

<!-- worked: vision_transformers.swin_stages -->
Swin-T on a 224 by 224 picture (Liu et al., 2021): patches of 4 pixels, widths 96, 192, 384, 768, blocks 2, 2, 6, 2, windows of 7 by 7 (the last stage's grid is one window). Patch merging between stages takes the grid and the width to a half and twice. FLOPs are the score and average products, 4 * pairs * width, of one block of the stage.

| stage | stride | grid | tokens | width | blocks | global (GFLOPs per block) | windowed (GFLOPs per block) | ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 56 x 56 | 3,136 | 96 | 2 | 3.776 | 0.059 | 64.0 |
| 2 | 8 | 28 x 28 | 784 | 192 | 2 | 0.472 | 0.030 | 16.0 |
| 3 | 16 | 14 x 14 | 196 | 384 | 6 | 0.059 | 0.015 | 4.0 |
| 4 | 32 | 7 x 7 | 49 | 768 | 2 | 0.007 | 0.007 | 1.0 |

Summed over the blocks of all four stages: 8.866 billion FLOPs if every block were global, 0.280 billion windowed, ratio 31.63.
<!-- /worked -->

Two things happen at once. The window turns the quadratic cost into a linear one, which matters most at the first stage: at stride 4 the grid is $56 \times 56$ and the windowed block is 64 times cheaper than a global one. And the merging shrinks the grid by 4 each stage, so by the last stage 49 tokens fit in one window and "windowed" and "global" are the same. Summed over all 12 blocks, the windowed network costs 0.280 billion FLOPs of attention products against 8.866 if every block were global, a ratio of 31.6.

The Qwen3-VL tower of brain applies a merge at the end instead of between stages: `PatchMerger` in `crates/qwen3vl/src/encoder.rs` normalizes, concatenates each $2 \times 2$ block of patch features (the patches arrive in "spatial-merge-block order", so the four of a block are contiguous and the gather is a free reshape, which is why `vision_position_ids` emits them that way), and maps them through two linear layers to the width of the language model. The tower attends over $T$ patches, and the language model sees $T/4$ tokens.

## Counting the whole network

The figures below put the designs side by side. For each network, the FLOPs of the score and average products over all blocks (not the projections or the feed-forward networks, which grow linearly with the tokens and are a separate bill), at four picture sides. The plain ViT is 12 blocks of width 768 and 16-pixel patches; the windowed variants use windows of $14 \times 14$ in every block and in the SAM arrangement of 8 windowed and 4 global blocks; Swin-T is as above; the hybrid is the encoder of the next section.

<!-- worked: vision_transformers.pyramid_table -->
Attention FLOPs, in billions, of the score and average products over the whole network (4 * pairs * width per block, projections and feed-forward networks not counted) at four picture sides. Plain ViT: 12 blocks of width 768, patches of 16 pixels. Windows 14 by 14: every block windowed, then the SAM arrangement of 8 windowed and 4 global blocks (the sides divide by 14 * 16). Swin-T as in its stage table, windows of 7 by 7. Hybrid encoder: the attention blocks of stages 3 and 4 at global attention, 4 and 2 blocks, widths 768 and 1536 (the sides divide by 64).

| design | side 448 | side 896 | side 1344 | side 1792 |
|---|---:|---:|---:|---:|
| plain ViT, global | 22.7 | 362.5 | 1,835.4 | 5,800.6 |
| plain ViT, windows 14 x 14 | 5.7 | 22.7 | 51.0 | 90.6 |
| plain ViT, windows with 4 global blocks | 11.3 | 136.0 | 645.8 | 1,994.0 |
| Swin-T, windows 7 x 7 | 1.1 | 4.5 | 10.1 | 17.9 |
| hybrid encoder (attention in two stages) | 0.5 | 8.0 | 40.6 | 128.4 |
<!-- /worked -->

![Attention FLOPs of the score and average products of the whole network against the side of the picture, both axes logarithmic. The plain ViT with global attention is the steepest line, rising from 2.3e10 at 448 pixels to 5.8e12 at 1,792; the windowed ViT, Swin-T and the hybrid encoder lie below it, and the line of windows with four global blocks lies between the plain ViT and the others.](img/attention-flops-resolution.svg)

The slopes tell the story. On the log axes, global attention in a plain ViT has slope 4 (the product grows as the fourth power of the side); the windowed networks have slope 2; the SAM arrangement starts with slope 2 and bends toward 4 as its four global blocks take over; Swin-T is windowed at three stages and its last stage is global but tiny. The hybrid encoder is global in its two attention stages and has slope 4 as well, but it runs 45.2 times below the plain ViT at every size, because its attention runs on grids with strides of 32 and 64 instead of 16 and in 6 blocks instead of 12.

![Tokens against the side of the picture, both axes logarithmic, four parallel lines. From the top: Swin-T's first stage at stride 4 (12,544 tokens at 448 pixels), the plain ViT with 16-pixel patches (784), Swin-T's last stage at stride 32 (196) and the hybrid encoder's output at stride 64 (49).](img/tokens-resolution.svg)

## Convolutions first, attention last: FastViTHD

A pyramid has a lesson for what to put where. At the high-resolution end the grid is large, every token covers a small piece of the picture, and what is needed is local mixing: edges and textures, the job of the convolutions of Chapter 2. At the low-resolution end the grid is small and the tokens are wide, and global attention is affordable and is what carries the context. A **hybrid** puts convolutional stages first and attention in the last stages. Brain's FastVLM crate implements one: the vision encoder FastViTHD in `crates/fastvlm/src/encoder.rs`, whose configuration is `FastVitHdConfig::fastvithd` (the design descends from FastViT, Vasu et al., 2023, and FastVLM, Vasu et al., 2025, uses it as a vision tower).

The encoder takes a $1024 \times 1024$ picture through a stem of three convolutions (a $3 \times 3$ stride-2 convolution from 3 to 96 channels, a depthwise $3 \times 3$ stride-2 convolution, a $1 \times 1$) to stride 4, then five stages with a downsample between each pair, of $[2, 12, 24, 4, 2]$ blocks, 44 in all, and widths $[96, 192, 384, 768, 1536]$. Stages 0 to 2 are convolutional, stages 3 and 4 use attention.

A convolutional block, called a **RepMixer** block, lets tokens exchange information with one depthwise $3 \times 3$ convolution and then passes each token through a **ConvFFN**, a feed-forward network that also looks at the neighbours: a depthwise $7 \times 7$ convolution with normalization, then two $1 \times 1$ convolutions that widen the channels by 4 and bring them back, with a GELU between them. An attention block keeps the ConvFFN and replaces the depthwise mixer by multi-head attention. Both are written out, with their scaling and folding, after the table.

<!-- worked: vision_transformers.fastvithd_stages -->
The encoder of `FastVitHdConfig::fastvithd` on a 1024 by 1024 picture: a stem of three convolutions to stride 4, then five stages at strides 4, 8, 16, 32 and 64 with a stride-2 downsample between them, 44 blocks in all. Multiply-adds (MACs) are counted from the layer shapes, weights only (no norms, biases or activations). The final expansion `conv_exp` is a 3 by 3 convolution with groups = 1,536 that doubles the 1,536 channels to 3,072 (9 weights per output channel) and its squeeze-and-excitation gate of 3,072 -> 192 -> 3,072 on the pooled vector (hidden width 3,072/16); each stage's RepMixer block is a depthwise 3 by 3 (9 c hw) and a ConvFFN of a depthwise 7 by 7 (49 c hw) and two 1 by 1 layers c -> 4c -> c (8 c^2 hw), an attention block adds 4 c^2 hw and 2 hw^2 c to the same ConvFFN, and a downsample is a depthwise 7 by 7 and a 1 by 1 at the output grid; the position convolutions are depthwise 7 by 7 before the two attention stages. "Token mixer" is what the block spends on letting tokens see each other: the fused depthwise 3 by 3 convolution (9 c hw) in stages 0 to 2 and the score and average products (2 hw^2 c) in stages 3 and 4; the next column is what global attention would cost there.

| stage | grid | tokens hw | width c | blocks | token mixer | mixer (GMACs per block) | global attention (GMACs per block) | GMACs of the stage's blocks |
|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 0 | 256 x 256 | 65,536 | 96 | 2 | RepMixer | 0.057 | 824.634 | 10.393 |
| 1 | 128 x 128 | 16,384 | 192 | 12 | RepMixer | 0.028 | 103.079 | 60.171 |
| 2 | 64 x 64 | 4,096 | 384 | 24 | RepMixer | 0.014 | 12.885 | 118.154 |
| 3 | 32 x 32 | 1,024 | 768 | 4 | attention | 1.611 | 1.611 | 35.588 |
| 4 | 16 x 16 | 256 | 1536 | 2 | attention | 0.201 | 0.201 | 14.937 |

In billions of multiply-adds: stem 1.340, the four downsamples 2.705, the two position convolutions 0.058, the final expansion 0.008; blocks 239.243. Total 243.354 billion multiply-adds, 486.708 billion FLOPs (1.7 percent outside the blocks). Exactly, in MACs: one attention block of stage 0 would need 824,633,720,832 for global attention, the depthwise convolution that is there 56,623,104.
<!-- /worked -->

The table says why. With the width doubling as the grid halves on each side, a $1 \times 1$ layer costs $c^2\,hw$ and is the same at every stage (the four times larger $c^2$ meets the four times smaller $hw$), so each convolutional block costs about 5 billion multiply-adds wherever it sits and the stages differ only in how many blocks they have. A token mixer does not behave that way. The depthwise convolution costs $9c\,hw$ and falls with the grid. Global attention costs $2hw^2c$ and *rises* as the grid grows: at stage 0 it would be 824.634 billion multiply-adds per block, against 0.057 billion (56,623,104) for the depthwise convolution that is there, a factor of 14,563.6. Attention is affordable at 1,024 tokens and 256, where it costs 1.611 and 0.201 billion; it is not at 65,536.

The blocks, as the crate builds them. A stage 0 to 2 block is a **RepMixer** block: a token mixer that at inference is one depthwise $3 \times 3$ convolution with a bias, then a ConvFFN (a depthwise $7 \times 7$ convolution with normalization, then two $1 \times 1$ convolutions with the GELU between them, expanding the width by 4), and a learned per-channel scale, the LayerScale, on the residual branch: `out = mixer(x) + layer_scale * ConvFFN(mixer(x))` (`RepMixerBlock::forward`). "At inference" matters. The training-time form of such a block has parallel branches (Vasu et al.), and the checkpoint that brain reads ships the fused inference form, the fold that Chapter 5 derived for ZipDepth: `crates/fastvlm/src/vision_import.rs` maps names like `token_mixer.reparam_conv` and `proj.0.lkb_reparam` onto brain's layout, and converts each LayerScale vector $\mathbf{s}$ to the pair $[\mathbf{s} - 1, \mathbf{0}]$ that its `film_chan` kernel wants (`y = x * (1 + scale) + shift`). So brain's FastViTHD is an inference graph with training-time structure already folded away. It also has a backward pass, built block by block with a gradient check for each (the subjects of the July 2026 commits of the crate list them: convolution unit, ConvFFN, RepMixer block, downsample, and the attention block with the encoder), and that differentiates the folded form.

The downsample between stages is a grouped $7 \times 7$ convolution of stride 2 with `groups = in_ch` that doubles the channel count, followed by a $1 \times 1$ convolution (`PatchEmbed`): a patch merging with a larger and depthwise kernel. Before each attention stage, `RepCPE` adds the positions, as the section on positions described. An attention block (`AttentionBlock`) is `x = x + ls1 * MHSA(LN(x))`, then `x = x + ls2 * ConvFFN(x)`: the map is transposed from channel-first to token rows, normalized, multiplied by one fused weight for the queries, keys and values, attended with heads of 32 channels and no mask, projected, and transposed back. It runs the full-attention routine of the shared ViT builder, `model::vit::chunked_attn_fwd`, so the tower adds no device kernel of its own; the crate header says so ("no net-new device kernels"). Commit `124b021fe` (July 2026), which added the block, says the same, and its test is a 64-channel block over a $4 \times 4$ map that stays finite and keeps its shape. The final layer, `conv_exp`, is a grouped $3 \times 3$ convolution that doubles the 1,536 channels to 3,072, with a squeeze-and-excitation gate.

By the same counts the whole encoder costs 243.354 billion multiply-adds (486.708 billion FLOPs) on a 1,024-pixel picture, 98.3 percent of them in the blocks.

Whether the tower is *right* is a different question from whether the counts are. The crate's test `fastvlm_vision_tower_matches_hf` runs the tower on the real checkpoint's weights and a reference implementation's preprocessed pixels and asserts a cosine similarity above 0.999 on the $[256, 3072]$ features; commit `79a005bd7` (July 2026) reports, from a run, a mean absolute difference of 3.6e-7 and a cosine of 1.000000. The test skips when the checkpoint or the reference dump is absent, so the number is a claim of the commit and not something the repository's default test run re-derives.

## How the tokens reach the language model

A hybrid encoder's output is a grid of vectors, and a language model reads a sequence. The path from picture to prompt in `crates/fastvlm/src/caps.rs`:

1. The picture is padded to a square with a grey of 0.5 (`image_aspect_ratio: "pad"` in the checkpoint's configuration) and then resampled bilinearly to $1024 \times 1024$. The order matters: pad first, resample after, so that the content blends into the border at the seam. Commit `0b5d89342` (August 2026) replaced a hand-written sampler in this function with the shared resize of `crates/imaging` for the reason Chapter 1 gave, and the function's comment notes the one numeric consequence, the blend at the seam. The tower, the projector and the decoder are each checked against a reference, and the pad-then-resample order is the checkpoint's own configuration.
2. The encoder produces $16 \times 16 = 256$ tokens of 3,072 numbers.
3. A projector, `mlp2x_gelu`, maps each token with a linear layer, a GELU and another linear layer, 3,072 to 896 to 896 for the 0.5-billion model, to the width of the language model.
4. The language model is Qwen2, run by the Qwen decoder of `crates/qwen3` with two toggles, no query-key normalization and biases on the query, key and value projections (`FastVlmConfig::decoder_of`). The 256 projected rows enter the prompt between the text before the picture (`<|im_start|>user\n`) and the text after it, as rows of embeddings instead of token ids (`PrefillInput::Embed` in `CaptionAction::run`).

The positions are the interesting part. The decoder uses plain one-dimensional RoPE (the module header of `crates/fastvlm/src/model.rs` says "no M-RoPE"). The 256 image tokens are a run of consecutive rows of the sequence, in row-major order, and the language model is not told that they were a $16 \times 16$ grid. Whatever two-dimensional structure the tokens carry, they got it inside the encoder, from convolutions, RepCPE and the pyramid. The Qwen3-VL model of brain does the opposite: its decoder uses the multi-axis rotary embedding of Chapter 8 of Volume V to place image tokens on three axes, and its vision tower applies two-axis RoPE as well. That is the design Chapter 9 takes up.

The budget is the number of tokens that the language model must process, which is the number its prompt grows by, and the encoder's own cost on top. Both are counted here for four picture sizes, with the decoder of FastVLM-0.5B.

<!-- worked: vision_transformers.token_budget -->
Tokens a language model reads for one square picture: a plain ViT with 16-pixel patches (T = (s/16)^2), a 16-pixel ViT whose patches are merged 2 by 2 before the language model (the Qwen3-VL arrangement, (s/32)^2), and the 64-fold reduction of the hybrid encoder ((s/64)^2). Encoder FLOPs in billions: twelve blocks of the base size (D = 768) at the plain token count, by the block count of the earlier table (patch embedding left out, but the norms, adds and GELU of the block counted, T (10 D + 6 + 5 F) FLOPs), against the hybrid encoder's stage table (2 FLOPs per multiply-add); these are different models, counted alike. The decoder of FastVLM-0.5B has 24 layers of width 896, 14 query and 2 key-value heads of width 64 and a feed-forward width of 4864; its weight multiply-adds per token, summed over the layers (attention 896 * 896 + 2 * 896 * 128 + 896 * 896, feed-forward 3 * 896 * 4864), are 357,826,560. The last two columns are twice that times the number of image tokens, in GFLOPs (the attention scores between tokens are not included).

| side (px) | tokens: plain / merged / hybrid | plain ViT-B/16 encoder | hybrid encoder | decoder, plain tokens | decoder, hybrid tokens |
|---|---:|---:|---:|---:|---:|
| 256 | 256 / 64 / 16 | 46.020 | 29.619 | 183.207 | 11.450 |
| 512 | 1,024 / 256 / 64 | 213.639 | 119.112 | 732.829 | 45.802 |
| 1024 | 4,096 / 1,024 / 256 | 1,327.472 | 486.708 | 2,931.315 | 183.207 |
| 1536 | 9,216 / 2,304 / 576 | 4,760.248 | 1,133.593 | 6,595.459 | 412.216 |
<!-- /worked -->

Read the table in two directions. In rows, the hybrid hands the language model 4 times fewer tokens than the merged ViT and 16 times fewer than the plain ViT with 16-pixel patches, at every size. In columns, the decoder's weight work for the image tokens at 1,024 pixels falls from 2,931.315 to 183.207 billion FLOPs, and the encoder's from 1,327.472 (twelve base blocks at 4,096 tokens) to 486.708. Added, the two come to 4,258.787 against 669.915 billion FLOPs, a factor of 6.4. Two cautions go with the number. The encoders are different models counted by the same rule, and the table says nothing about which gives the better answers. And the hybrid's own encoder costs 2.7 times what its decoder spends on the image tokens: the tokens are few, and the tower that produces them is not cheap. The attention between tokens, which the table leaves out, would add to the decoder's side; a timed comparison at a fixed accuracy is not in this repository.

The towers of brain, with the numbers above:

<!-- worked: vision_transformers.brain_towers -->
Image encoders of brain's crates: the size of the grid at the point where attention runs last, the tokens the encoder hands on, how a token learns where it is and where attention is local. The class token of CLIP is counted in its 257 and 577. The last column is what reaches the next stage: the SAM tower's two stride-2 convolutions after its neck take 4096 tokens to 256, and the Qwen3-VL merger folds each 2 by 2 group of patches into one token.

| tower | patch (px) | input (px) | grid | tokens in the encoder | positions | attention | tokens handed on |
|---|---|---:|---:|---:|---|---|---:|
| SAM ViT-B tower (`crates/sam1`) | 16 | 1024 | 64 x 64 | 4,096 | learned table (bicubic resample for another grid) and a relative bias | windows 14 x 14, blocks 2, 5, 8, 11 global | 256 |
| CLIP-L/14 body (`crates/clip`) | 14 | 224 | 16 x 16 | 257 | learned table, bicubic resample | global | 257 |
| EVA02-CLIP-L/14 (`crates/clip`) | 14 | 336 | 24 x 24 | 577 | two-axis rotary embedding | global | 577 |
| Qwen3-VL ViT, a 448 px picture (`crates/qwen3vl`) | 16 | 448 | 28 x 28 | 784 | learned 48 x 48 table, bilinear resample, two-axis rotary embedding | global | 196 |
| FastViTHD (`crates/fastvlm`) | 4, then 2 four times | 1024 | 16 x 16 | 256 | a depthwise 7 x 7 convolution before each attention stage | global, in the last two of five stages | 256 |
<!-- /worked -->

## What the convolution knew

A convolution comes with two assumptions built in. It looks only at a small neighbourhood (locality) and it applies the same filter everywhere (translation equivariance, Chapter 2). A vision transformer has neither: after a patch embedding it is a set processor with positions added, and attention can reach anywhere from the first layer. Everything a convolutional network gets for free, a transformer has to learn from data.

The reported findings about this are about scale. Dosovitskiy et al. (2021) found that a vision transformer trained on a mid-sized dataset (ImageNet) did worse than a comparable residual network, and that with pre-training on much larger datasets (tens to hundreds of millions of images) it matched and passed it. Touvron et al. (2021) showed that the ImageNet-only gap could largely be closed by a stronger training recipe (heavier augmentation and regularization) together with distillation from a convolutional teacher through an added token. These are results about the ImageNet scale and the recipes of those papers. This chapter does not reproduce them, and nothing below should be read as a test of them.

What a small experiment can show is the direction in which the regime goes when there is little data. The toy pictures of Chapter 3 (stripes, checkerboards and blobs, 60 for training and 90 for testing) are small enough for a vision transformer to be written out. The model: 2 by 2 patches (16 tokens of 4 numbers), the embedding to $D = 8$, a learned position table, one pre-norm block of Volume V without the mask, a final normalization, a readout and a linear classifier.

<!-- worked: vision_transformers.vit_setup -->
The images and the split are those of the convolutions chapter (`cnns.toy_data`: 150 images of 8 by 8 pixels, class n mod 3 for stripes, checkerboards and blobs; the first 60 train, the other 90 test). The model: 2 by 2 patches (16 tokens of 4 pixels, flattened row by row), a linear embedding to D = 8 with a bias, a learned position table of 16 rows added, one pre-norm block of Volume V (RMS norm, 2 heads of width 4 with no mask (head h takes columns 4h to 4h + 3 of the query, key and value projections, and the head outputs are concatenated in head order before the output projection), residual, RMS norm, a GELU feed-forward network of hidden width 16 without biases, residual), a final RMS norm, a readout, and a linear map to the 3 classes. Readout "mean": the mean of the 16 token rows. Readout "cls": a learned class row placed before the patches (no position row of its own), whose output row is read. The RMS norms use epsilon 1e-6 and the GELU is the exact one (with `erf`); the initialization is given with the training table below. Multiply-adds per image (patch embedding 16 * 4 * 8, four projections 4 * 16 * 8^2, the score and average products 2 * 2 * 16^2 * 4, the feed-forward network 2 * 16 * 8 * 16, the head 8 * 3): 12,824 for the mean readout, against 5,778 for the convolutional network.
<!-- /worked -->

The backward pass of the block is the chain of backward functions the previous chapters checked (the RMS norm, the feed-forward network, the attention), plus the embedding and the readout. As Volume III taught, the first thing to do with a hand-written backward pass is to check it:

<!-- worked: vision_transformers.vit_gradient_check -->
The initialized transformer (run 0) on the training images n = 0 .. 5 (two of each class): the analytic gradient of the mean cross-entropy against the central difference (L(w + h) - L(w - h)) / (2h) with h = 1e-6 in 64-bit arithmetic, one weight at a time, with the gate |analytic - numeric| <= 1e-7 + 1e-5 max(|analytic|, |numeric|) of Volume III, Chapter 6.

| readout | parameters checked | largest absolute difference | pass the gate |
|---|---:|---:|---:|
| mean | 731 | below 1e-7 | 731 |
| cls | 739 | below 1e-7 | 739 |

The class-token run checks the class token as well, so it has 8 parameters more.
<!-- /worked -->

Then training, with the same optimizer as the convolutional network of Chapter 3:

<!-- worked: vision_transformers.vit_training -->
Both readouts, run 0 (the generator of the name vision_transformers.vit_init.0, seed 0x19862e4db422479e (1839208409096931230); run r uses the name vision_transformers.vit_init.r), trained on the 60 training images by full-batch Adam exactly as in the convolutions chapter (beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8, learning rate 0.02, 100 updates, mean cross-entropy, sums over images in index order). Initialization: every entry is `normal() * scale` from that one generator, matrix by matrix and row by row, in this order: We [4, 8] with scale 0.5, the position table E [16, 8] with 0.3, Wq, Wk, Wv and Wo [8, 8] with 1/sqrt(8) each, W1 [8, 16] with 1/sqrt(8), W2 [16, 8] with 1/4, the classifier Wh [8, 3] with 1/sqrt(8), and last the class token [1, 8] with 0.3 (drawn, and unused, by the mean readout, so the two readouts share every other starting weight). The patch bias and the classifier bias are 0 and the three norm gains are 1. The nudge is added to We[0][0], the first weight of the patch embedding, which every image uses. A number that comes from a chaotic run is printed with only the digits that are the same in the run and in four copies of it whose first weight is changed by +1e-12, -1e-12, +3e-12 and -3e-12: in full if all agree on it, as 'about x' with the leading digits on which all agree, or as 'unstable' when not even one digit survives; the other digits cannot be reproduced in another implementation.

| readout | updates t | training loss | training accuracy |
|---|---:|---:|---:|
| mean | 0 | 1.2182 | 0.3167 |
| mean | 25 | 0.1171 | 0.9833 |
| mean | 50 | 0.0034 | 1.0000 |
| mean | 100 | 0.0004 | 1.0000 |
| cls | 0 | 1.4692 | 0.2500 |
| cls | 25 | 0.0972 | 0.9833 |
| cls | 50 | 0.0015 | 1.0000 |
| cls | 100 | 0.0003 | 1.0000 |
<!-- /worked -->

Run 0 of each readout reaches a training accuracy of 1.0 within 50 updates, and every run has it after 100. That is a statement about the training set, and the test set decides the matter:

<!-- worked: vision_transformers.vit_against_cnn -->
After 100 updates on the same 60 training images, scored on the same 90 test images (30 of each class), same optimizer and rate. The vision transformer is trained from three initializations (runs 0, 1, 2, generators of the printed seeds), each with both readouts; the convolutional network is the single run of the previous chapter. A number that comes from a chaotic run is printed with only the digits that are the same in the run and in four copies of it whose first weight is changed by +1e-12, -1e-12, +3e-12 and -3e-12: in full if all agree on it, as 'about x' with the leading digits on which all agree, or as 'unstable' when not even one digit survives; the other digits cannot be reproduced in another implementation.

| model | run (seed) | parameters | training accuracy | test cross-entropy | test accuracy |
|---|---|---:|---:|---:|---:|
| convolutional network (Chapter 3) | - | 283 | 1.0000 | 0.0997 | 0.9556 |
| vision transformer, mean | 0 (0x19862e4db422479e) | 731 | 1.0000 | 0.3856 | 0.8556 |
| vision transformer, mean | 1 (0x97992191bb0f964f) | 731 | 1.0000 | 0.6609 | 0.8222 |
| vision transformer, mean | 2 (0xf20a07d74527d871) | 731 | 1.0000 | 1.1962 | 0.7556 |
| vision transformer, cls | 0 (0x19862e4db422479e) | 739 | 1.0000 | 2.1916 | 0.6778 |
| vision transformer, cls | 1 (0x97992191bb0f964f) | 739 | 1.0000 | 2.0309 | 0.6889 |
| vision transformer, cls | 2 (0xf20a07d74527d871) | 739 | 1.0000 | 2.4469 | 0.6111 |
<!-- /worked -->

The convolutional network, with 283 parameters, scores 95.6 percent on the 90 test images (86 right). The transformers, with 731 or 739 parameters, fit the 60 training images perfectly and score 75.6 to 85.6 percent with the mean readout and 61.1 to 68.9 percent with the class token across the three initializations. The gap is not noise of the initialization: in each of the three runs the mean readout beats the class token by 13 to 18 points and the best transformer is 10 points under the convolutional network. That is the direction the cited findings predict for a small dataset, and it is all this experiment shows. It is one dataset of three classes, 60 images, one block, the learning rate of the convolutional network, not tuned for the transformer, 100 updates and three initializations. A larger or longer-trained transformer, a different patch size or a tuned recipe might narrow the gap; the experiment does not say. The readout comparison is as narrow: in this setting, with 60 images, averaging the patches is easier to fit than learning a class token that has to find its way to the patches, which is a plausible mechanism and not one that was isolated here.

The convolutional network of Chapter 3 has its prior in the architecture: a filter that sees a $3 \times 3$ neighbourhood and is shared everywhere, so it needs few examples to learn that a stripe is a stripe wherever it is. The transformer's table of positions has 128 free numbers and its attention can single out a patch at a particular position; a plausible reading, which this experiment did not isolate, is that on 60 images that is easier to memorize than to generalize. Hybrids like FastViTHD are one answer: let convolutions supply the prior where the grid is large, and attention take over where it is small.

## Where it lives in brain

Vision attention in brain is shared, in two builders. `crates/model/src/vit.rs` holds a pre-norm bidirectional block with optional per-head normalization of queries and keys, optional two-axis RoPE and optional LayerScale, and the span-wise attention that runs global, windowed, shifted and padded-window layouts from a list of spans; the Qwen3-VL tower and the FastViTHD attention block are built on it, and the SAM tower takes its window plan from it. The SAM tower runs its per-span attention through the other builder, `model::block` (`chunked_bidir_fwd`); the EVA02 tower composes its blocks from the primitives of `model::block`, because the shared ViT block has no hook for its extra normalizations (the header of `crates/clip/src/model.rs` explains); and the FastVLM crate adds the convolution blocks of `crates/vision`. The reuse is deliberate and has a history: the commit that added windows to the shared builder (`80178df9a`) says that Hiera, Swin and DaViT should compose from it instead of each porting its own attention, and the FastViTHD attention block went through the same attention function a few days earlier (`124b021fe`).

## Check your understanding

1. **Find the error.** "A $16 \times 16$ patch embedding of a $224 \times 224$ picture has $196$ times fewer parameters than a $1 \times 1$ embedding of every pixel, because there are 196 times fewer tokens." What is wrong, and what is the parameter count of each?
2. **Predict.** The patch matrix of a trained model is used with the pixels flattened in the order (row, column, channel) instead of (channel, row, column). What is the output, and would it look broken?
3. **Explain.** Why does a depthwise $7 \times 7$ convolution before an attention stage give the tokens a sense of position, when it has the same weights at every location?
4. **Find the error.** "Windowed attention with windows of $7 \times 7$ makes the cost of a $112 \times 112$ grid 16 times lower than global attention, because the window has 16 times fewer tokens than the grid." Give the right count.
5. **Predict.** A model is trained with a learned table on a $14 \times 14$ grid and used on $28 \times 28$ with the table resampled by bilinear interpolation with aligned corners, but the checkpoint was trained with the half-pixel bicubic rule. What happens to the tokens at the border?
6. **Explain.** Why can the ragged windows of a shifted partition run with no mask, and what do they save relative to the rolled windows with a mask?
7. **Find the error.** "A hybrid encoder with attention in the last two stages is cheaper than a plain ViT because convolutions are cheaper than attention." Give the correct reason, using the stage table.
8. **Explain.** In the experiment of this chapter, the transformer reaches a training accuracy of 1.0 and a test accuracy of 75.6 to 85.6 percent. What does that prove about transformers on images, and what does it not?

## Summary

A vision transformer cuts the picture into patches, embeds each by one matrix (which is a strided convolution), adds a position, and runs the unmasked block of Volume V. The patch makes the token count tractable: $T = (S/P)^2$, with attention costing $T^2$ pairs, so that doubling the side multiplies it by 16. Position on a grid is a table, a pair of sinusoids, a two-axis rotation or the side effect of a convolution; a table has to be resampled when the grid changes, and the resampling convention is part of the checkpoint. Windows make attention linear in the tokens, shifts or a few global blocks connect them, and merging patches between stages makes a pyramid with fewer tokens and wider channels as it goes, like a convolutional network's. A hybrid uses convolutions where the grid is large and attention where it is small, and hands the language model few tokens, which is the budget that matters to Chapter 9.

The limits: the counts are counts, of FLOPs and bytes, and no timing is claimed; the experiment is a toy, and says nothing about ImageNet; the hybrid's advantage in total cost is argued from counts, not measured. The next chapter takes the vision transformer that this one built and uses it for a task a classifier cannot do: segmenting whatever the user points at.

## References

- A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit and N. Houlsby, *An image is worth 16x16 words: transformers for image recognition at scale*, ICLR 2021. The patch embedding, the class token, learned positions, and the findings on data scale.
- H. Touvron, M. Cord, M. Douze, F. Massa, A. Sablayrolles and H. Jegou, *Training data-efficient image transformers and distillation through attention*, ICML 2021. A training recipe and a distillation token that let a vision transformer train on ImageNet alone.
- Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin and B. Guo, *Swin Transformer: hierarchical vision transformer using shifted windows*, ICCV 2021. Windowed attention, shifted windows with a mask, patch merging and the four-stage pyramid.
- P. K. A. Vasu, J. Gabriel, J. Zhu, O. Tuzel and A. Ranjan, *FastViT: a fast hybrid vision transformer using structural reparameterization*, ICCV 2023. The RepMixer block and the hybrid design that the encoder of this chapter follows.
- P. K. A. Vasu et al., *FastVLM: efficient vision encoding for vision language models*, CVPR 2025. The hybrid encoder used as the vision tower of a language model.
- X. Zhai, B. Mustafa, A. Kolesnikov and L. Beyer, *Sigmoid loss for language image pre-training*, ICCV 2023. The SigLIP vision-language model whose tower has no class token.

## Exercises

1. **[Calculate]** A picture of $512 \times 512$ pixels is cut into patches of 8 and of 32 pixels. Give the tokens, the entries of one head's score matrix, and the FLOPs of the score and average products for $D = 768$, in each case.
2. **[Calculate]** A patch embedding has $C = 3$, $P = 14$ and $D = 1024$. Give its parameters and the length of the flattened patch, and the number of patches in a picture of $336 \times 336$ pixels.
3. **[Calculate]** For a $32 \times 32$ grid of tokens and windows of $8 \times 8$, give the global and windowed pair counts and their ratio. With a shift of 4 and ragged windows, give the pair count as the product of two axis sums.
4. **[Calculate]** Resample the table $[0, 1, 2, 3]$ (one row, one channel) to 8 values by the half-pixel rule with bilinear interpolation and by cubic convolution with $a = -0.75$ and clamped taps, and compare the two at the first and the last output.
5. **[Calculate]** For the hybrid encoder of the chapter at $512 \times 512$, give the grid, the tokens and the width of each stage, and the multiply-adds of one attention block in stage 3.
6. **[Calculate]** Swin-T's four stages have 2, 2, 6 and 2 blocks at widths 96, 192, 384 and 768. For a $448 \times 448$ picture give the windowed attention products per stage and in all.
7. **[Implement]** Reproduce `vision_transformers.patches` and `vision_transformers.embedding_equivalence`: the 16 flattened patches, and the matrix and convolution forms of the embedding to the stated criterion, including the difference caused by flattening in the other order.
8. **[Implement]** Reproduce `vision_transformers.grid_offsets`, `vision_transformers.rope_2d_scores` and `vision_transformers.position_resample`: the two-axis and one-axis sinusoid tables, the scores of rotated queries and keys at the listed offsets, and the 6 by 6 table by the clamped cubic with $a = -0.75$ with the largest differences from the antialiased and the corner-aligned rules.
9. **[Implement]** Reproduce `vision_transformers.shifted_window_mask`: the rolled token ids, the region labels, the mask of each window and the agreement of the rolled and ragged attention outputs to the stated criterion.
10. **[Implement]** Reproduce `vision_transformers.fastvithd_stages` and `vision_transformers.token_budget` from the layer lists described in the text: the multiply-adds of each stage and the tokens and decoder work of each picture size.
11. **[Implement]** Reproduce `vision_transformers.vit_gradient_check` and the run 0 rows of `vision_transformers.vit_training` (the generator, the initialization order, the block built from the earlier chapters' functions, full-batch Adam), to the digits that survive the nudges.
12. **[Investigate]** Train the toy transformer with the patch size 4 (4 tokens) and with patch size 1 (64 tokens), keeping everything else, and report the test accuracy of each readout against the patch-2 result. Which way does it move, and what does the cost per image do?

## Key terms

- **patch embedding**: the layer that turns a picture into a sequence of tokens by cutting it into non-overlapping square patches, flattening each and applying one shared linear map; identical to a convolution whose kernel size equals its stride. <!-- concept: patch-embedding -->
- **vision transformer**: a model that applies a stack of unmasked transformer blocks to a sequence of patch embeddings with position information added, and reads out a class token or the mean of the tokens. <!-- concept: vision-transformer -->
- **windowed attention**: attention in which each token reads only the tokens in its own non-overlapping window of the grid, so that the cost grows linearly with the number of tokens; shifting the windows between layers, or adding a few global layers, lets information cross the window borders. <!-- concept: windowed-attention -->
- **hierarchical vision transformer**: a vision transformer arranged as a pyramid of stages, in which merging neighbouring patches between stages reduces the number of tokens and widens the channels, like the stages of a convolutional network. <!-- concept: hierarchical-vision-transformer -->
- **Swin transformer**: a hierarchical vision transformer whose attention is confined to non-overlapping windows and whose windows are displaced by half a window in alternate layers, so that information crosses window borders at the same cost per layer. <!-- concept: swin-transformer -->
- **RepMixer**: a convolutional token-mixing block of a hybrid vision encoder, trained with parallel branches and folded at inference into one depthwise $3 \times 3$ convolution, followed by a ConvFFN. <!-- concept: repmixer -->
- **ConvFFN**: a feed-forward network for feature maps: a depthwise $7 \times 7$ convolution with normalization, then two $1 \times 1$ convolutions that widen the channels by a factor (4) and narrow them again with a GELU between them. <!-- concept: convffn -->

## Answers

**Recall questions.**

1. $T^2$ entries per head. The products cost $4T^2D$ FLOPs for the block: $2T^2D$ for the scores and $2T^2D$ for the weighted average, over all heads.
2. The output is $[4, 4, 4]$: $(8 - 2)/2 + 1 = 4$ on each axis. The weights are $4 \cdot 3 \cdot 2 \cdot 2 = 48$ (plus 4 biases), and the multiply-adds are $16 \cdot 48 = 768$.
3. The output rows come out permuted in the same way, so the model cannot tell the order. It is equivariant to permutations.
4. Flattened with $w$ patches per row, the offset is $w$ (3 in a picture 3 patches wide, 5 in one 5 wide). Multi-axis RoPE gives $(1, 0)$ on the row and column axes whatever the width.
5. $(0 + 0.5) \cdot 4/6 - 0.5 = -1/6$, outside the picture, which is why the border rule matters.
6. A query reads $w$ keys (itself and $w - 1$ before it, or the first $i + 1$ for the first positions). The pair count is $wT$ for $T \gg w$, linear in $T$.

**Check your understanding.**

1. The parameters of the embedding do not depend on the number of tokens: they are $D\,CP^2 + D$ whatever the picture. For $D = 768$ and $C = 3$ the $16 \times 16$ embedding has $768 \cdot 768 + 768 = 590{,}592$ parameters, and a per-pixel embedding $768 \cdot 3 + 768 = 3{,}072$. The patch embedding has more parameters; what it saves is tokens.
2. The output is a different number for every patch: the matrix columns are in the order the model was trained with, so the output is a matrix product of the wrong pixels with each column. It would look like a plausible vector, finite and of the right shape, and the features would be wrong (the worked block shows differences of the order of the tokens themselves).
3. Zero padding. A token at the border sees zeros where a token in the interior sees pixels, so the output of the convolution differs between the border and the interior even with the same weights, and in a deeper layer the difference propagates inward. Position is thus encoded implicitly, relative to the border.
4. There are $(112/7)^2 = 256$ windows, and the saving is $T/M^2 = 112^2/49 = 256$ times, not 16. The window has 49 tokens, and the grid $12{,}544$, so the ratio of tokens is 256, which is the ratio of pairs.
5. The tokens at the border read positions that differ from the ones trained with, by up to the amount the two rules disagree there (the block shows 0.704 on a toy table); the output is plausible and wrong. It would not appear in a shape test.
6. The ragged windows are exactly the groups of tokens that the mask lets attend; so a span-wise attention over the groups is the masked attention. They compute only the allowed pairs (576 instead of 1,024 in the example) and need no masked softmax kernel.
7. Convolution is not cheaper per se: a $1 \times 1$ layer costs the same at every stage. The saving is that attention would cost $2hw^2c$ at stage 0 (824,633,720,832 multiply-adds per block) against 56,623,104 for the depthwise convolution that is there. The reason is the quadratic growth of attention in the number of tokens, and the hybrid avoids it by running attention on 1,024 and 256 tokens.
8. It proves that, with 60 images, one block, one tuned-for-the-CNN rate and 100 updates, this transformer memorizes the training set and generalizes worse than the convolutional network. It does not show that transformers need more data at ImageNet scale, that a tuned transformer would not close the gap, or anything about the readout beyond this setting.

**Exercises** (selected).

1. Patch 8: $T = 64^2 = 4{,}096$, $16{,}777{,}216$ entries, $4 \cdot 4{,}096^2 \cdot 768 = 51{,}539{,}607{,}552$ FLOPs. Patch 32: $T = 16^2 = 256$, $65{,}536$ entries, $4 \cdot 256^2 \cdot 768 = 201{,}326{,}592$ FLOPs.
2. $3 \cdot 14^2 = 588$ numbers per patch, so $1024 \cdot 588 + 1024 = 603{,}136$ parameters, and $(336/14)^2 = 576$ patches.
3. Global: $1024^2 = 1{,}048{,}576$; windowed: $1024 \cdot 64 = 65{,}536$, ratio 16. Shifted by 4 with windows of 8 on a grid of 32: axis cuts of lengths $4, 8, 8, 8, 4$, so $\sum \ell^2 = 16 + 64 + 64 + 64 + 16 = 224$ and the pair count is $224^2 = 50{,}176$.
4. Bilinear half-pixel at output 0: source $-0.25$, clamped to 0 (value 0); at output 7: source $3.25$, clamped to 3 (value 3). Cubic with $a = -0.75$ and clamped taps: output 0 has source $-0.25$, so $\lfloor\xi\rfloor = -1$, $t = 0.75$ and taps $-2, -1, 0, 1$ clamped to $0, 0, 0, 1$ with weights $-0.0352, 0.2617, 0.8789, -0.1055$, a value of $-0.1055$; the last output is symmetrical, $3.1055$. The cubic overshoots the range of the table at the borders and the bilinear rule cannot.
5. Grids $128, 64, 32, 16, 8$ at strides $4, 8, 16, 32, 64$, tokens $16{,}384, 4{,}096, 1{,}024, 256, 64$, widths $96, 192, 384, 768, 1536$. Stage 3 at 512 pixels has $T = 256$ and $c = 768$: one attention block costs $4c^2T = 603{,}979{,}776$ for the projections, $2T^2c = 100{,}663{,}296$ for the products and $Tc(49 + 8c) = 1{,}217{,}593{,}344$ for the ConvFFN, in all $1{,}922{,}236{,}416$ multiply-adds.
6. At 448: grids $112, 56, 28, 14$; windowed products $4 \cdot T \cdot 49 \cdot c$ per block: $4 \cdot 12{,}544 \cdot 49 \cdot 96 = 236{,}027{,}904$, $4 \cdot 3{,}136 \cdot 49 \cdot 192 = 118{,}013{,}952$, $4 \cdot 784 \cdot 49 \cdot 384 = 59{,}006{,}976$, $4 \cdot 196 \cdot 49 \cdot 768 = 29{,}503{,}488$; times $2, 2, 6, 2$ blocks gives $472{,}055{,}808 + 236{,}027{,}904 + 354{,}041{,}856 + 59{,}006{,}976$, a total of $1{,}121{,}132{,}544$, which is the value in the table for side 448.
