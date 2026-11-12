<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Generating Video

*How do diffusion models keep motion consistent across hundreds of frames?*

In August 2026 a real generation of 25 frames at 1920 by 1088 pixels came back correct for its first seven tenths of a second and then warped and smeared over its last several frames. The same prompt, the same still, the same seed and the same code at 1280 by 720 and at 512 by 512 were clean. Nothing was wrong with any single frame: each was a plausible picture, no value was not-a-number, and no per-frame statistic left its range. What had failed was the agreement between frames, which is the one thing a video has and a stack of photographs does not. Commit `d679b3eae` records the hunt for the cause, and it is worth following because it touches nearly everything this chapter explains: the number of tokens the transformer attends over, the schedule of noise levels it was taught to denoise, and the way a finished piece of one clip is handed to the next.

The first question is how to notice such a defect without watching the clip. The repository's answer is a curve, the mean absolute difference between each frame and the next, taken after every frame is averaged down to a 128 by 128 grid so that the numbers do not depend on resolution (`frame_to_frame_diffs` in `crates/ltxv/src/clipmetric.rs`), and one number from it, `blowup_ratio`, the largest difference divided by the median one. Footage with ordinary motion, fast or slow, keeps the ratio near 1 because the median follows the clip's own pace. A clip that disintegrates part of the way through pushes it far above. The table runs the arithmetic on four clips so small that every frame is one number: the probe of a constant frame is its value, and a difference is $|a_{i+1} - a_i|$ times 127.5.

<!-- worked: video_generation.clip_stability -->
Four clips of 16 constant frames, a frame being the single number a in [-1, 1] that every pixel holds, shown as (a + 1) * 127.5. Clip 1: a_i = -0.6 + 0.03 i. Clip 2: a_i = -0.9 + 0.1 i. Clip 3: a_i = -0.6 + 0.03 i for i = 0 .. 11, then 0.7, -0.7, 0.7, -0.7. Clip 4: a_i = -0.6 + 0.03 i for i = 0 .. 7, then a_i = 0.2 + 0.03 (i - 8) for i = 8 .. 15. The difference d_i is |a_(i+1) - a_i| * 127.5 for i = 0 .. 14; the median is element 7 of the 15 sorted differences (the upper of the middle pair when the count is even); the ratio is the largest difference over that median.

| clip | differences | median | largest | ratio |
|---|---:|---:|---:|---:|
| slow steady drift | 15 | 3.825 | 3.825 | 1.00 |
| fast steady drift | 15 | 12.750 | 12.750 | 1.00 |
| steady, then four thrashing frames | 15 | 3.825 | 178.500 | 46.67 |
| steady with one hard cut | 15 | 3.825 | 75.225 | 19.67 |
<!-- /worked -->

The fourth row is the honest limit of the instrument. A legitimate cut and a collapse look the same to it, and a clip that is the wrong video but steady scores a perfect 1. It is a stability gate, as its own module comment says, and it is no judge of quality. Its use here is as a tripwire: with everything but the resolution held fixed, the commit measured the ratio at 1.06, 1.03, 1.04 and 1.04 for clips of 1024, 2040, 3520 and 5600 video tokens, and at 14.66 for 8160. Only the last frame of the latent had gone wrong, and decoding the bad latent with its last frame replaced by its neighbour took the ratio from 17.43 to 1.31, which cleared the decoder and pointed at the transformer. The rest of the chapter is the machinery that makes consistency possible and the places where it can break. A clip is kept consistent inside one pass by attention that reaches across time as freely as across space, by positions that mean the same thing in every clip, and by a schedule fitted to the number of tokens; it is kept consistent from one pass to the next by conditioning, tokens that are already finished when the next pass begins.

**Video diffusion** is the generation procedure of Volume VIII run on the latent grid of the [previous chapter](01-video.md): draw a grid of noise with a time axis, let a transformer predict the velocity of every token given the prompt, and integrate with a sampler until the grid is a clip's latent, which the autoencoder decodes. Nothing in that sentence is new. What is new is that the grid has three axes, that its tokens are many and bound together by time, and that some of them may already be known. Five ideas from earlier volumes are needed first; the answers are at the end.

1. (Volume X, [Chapter 1](01-video.md), the latent grid.) How many latent frames does a clip of 121 frames have, and of 25 frames? What happens to a request for 24 frames?
2. (Volume V, [Chapter 7](../v05-sequences/07-self-attention.md), self-attention.) How many scores does one head of self-attention compute for $N$ tokens, and how many FLOPs do the scores and the weighted sum of values take for tokens of width $d$ (a multiply-add counts as two FLOPs)?
3. (Volume V, [Chapter 8](../v05-sequences/08-positions.md), multi-axis rotary embedding.) A head of width 128 has its channels divided among the three axes of a patch grid. How many rotation pairs does it have, and what rotates a pair?
4. (Volume VIII, [Chapter 6](../v08-image-generation/06-flow-matching.md), the timestep shift.) With a shift of $k = 3$, what does the noise fraction $\sigma = 0.5$ become, and what is the strength $\mu$ of that shift?
5. (Volume II, [Chapter 2](../v02-learning/02-linear-regression.md), least squares.) What does the least-squares line through $c$ noisy points estimate that a single point cannot?

## Attention that reaches across time

A latent frame of a 768 by 512 clip is a grid of 16 by 24 tokens, 384 of them, and a clip of $F'$ latent frames is $F' \times 384$ tokens in one sequence, laid out frame by frame, row by row, column by column. The 121-frame clip has 16 latent frames and 6144 tokens. What the transformer does with those tokens decides whether the frames can agree. If each latent frame were denoised alone, with attention only inside it, the clip would be 16 unrelated pictures. Letting a token look at tokens of other frames is what lets a ball that leaves the left edge of one frame be the same ball at the right edge of another, and the design question is how many of the possible pairs to allow.

**Spatiotemporal attention** is self-attention in which the tokens of a clip, placed on a grid of frame, row and column, attend to one another across all three axes. There are three ways to spend the pairs. The cheapest is frame by frame, $F' N_{\mathrm f}^2$ pairs for $N_{\mathrm f}$ tokens in a latent frame, which is no attention across time at all. The middle way alternates layers: in one layer a token attends within its frame, in the next along its own position through time, $F' N_{\mathrm f}^2 + N_{\mathrm f} F'^2$ pairs over the two layers. The dearest is all at once, $(F' N_{\mathrm f})^2$ pairs, every token against every token. The table counts pairs for one head of one layer of a clip at 768 by 512.

<!-- worked: video_generation.attention_pairs -->
The latent frame of a 768 x 512 clip is 16 x 24 = 384 tokens. For a clip of F pixel frames (F = 1 + 8k) with F' = 1 + k latent frames and S = 384 tokens in each, one head of one attention layer scores: F' S^2 pairs when every latent frame attends only within itself; F' S^2 + S F'^2 when each layer is either spatial (within a frame) or temporal (within a position, across frames); (F' S)^2 when every token attends to every token.

| F | F' | tokens | frame by frame | space then time | all at once | all at once / space then time |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 384 | 147,456 | 147,840 | 147,456 | 1.0 |
| 9 | 2 | 768 | 294,912 | 296,448 | 589,824 | 2.0 |
| 25 | 4 | 1,536 | 589,824 | 595,968 | 2,359,296 | 4.0 |
| 49 | 7 | 2,688 | 1,032,192 | 1,051,008 | 7,225,344 | 6.9 |
| 97 | 13 | 4,992 | 1,916,928 | 1,981,824 | 24,920,064 | 12.6 |
| 121 | 16 | 6,144 | 2,359,296 | 2,457,600 | 37,748,736 | 15.4 |
| 241 | 31 | 11,904 | 4,571,136 | 4,940,160 | 141,705,216 | 28.7 |
| 481 | 61 | 23,424 | 8,994,816 | 10,423,680 | 548,683,776 | 52.6 |

One picture (F = 1) costs 147,456 pairs; the 121-frame clip costs 37,748,736 with all-at-once attention, 256 times as many (16 frames, so 16 times the tokens and 16 squared the pairs).
<!-- /worked -->

![Pairs of positions scored by one attention head against the number of tokens of a 768 by 512 clip, on logarithmic axes: attention frame by frame and attention that alternates space and time rise together with slope one, and attention over all tokens at once rises with slope two, 15 times higher at 6144 tokens.](img/attention-pairs.svg)

The last column is the point. Alternating space and time keeps the cost close to linear in the number of frames, and all-at-once attention grows with their square: 15.4 times dearer at 16 latent frames, 52.6 times at 61. The 121-frame clip costs 256 times the pairs of a single picture, 16 times the tokens and 16 squared the pairs, and the minute of video of the last chapter's last table, 181 latent frames and 92 672 positions, would cost $181^2 = 32\,761$ times the pairs of one frame. What the dear way buys is reach. With alternating layers, a token at $(f, r, c)$ can learn about the token at $(f + 1, r + 1, c + 1)$ only through a chain of layers, one for each axis on which the two differ, while all-at-once attention reaches it in one. A scene in which an object moves diagonally across the frame is where the difference shows. Both of brain's video models pay for reach. `attn1` of every block of `crates/ltxv/src/block.rs` attends over all tokens of the window, and the module comment of `crates/ltxv/src/longform.rs` leans on it ("its attention is global over the whole window, so a frozen prefix is visible to every generated token"). `crates/wan/src/lib.rs` says the same of Wan: each block runs self-attention over the flattened latent volume. Neither crate implements a space-then-time variant; the middle column is for comparison.

The prompt reaches the tokens a different way from the joint sequence of Volume VIII's [denoiser chapter](../v08-image-generation/08-denoiser-architectures.md). Both video models give each block a separate cross-attention from the video tokens to the prompt's rows, the topology of the SDXL U-Net, which the Wan crate's own comment names, and which adds $N \times N_{\mathrm{ctx}}$ scores, where appending the prompt to the sequence of a joint design would score $(N + N_{\mathrm{ctx}})^2$. In LTX the context is 1024 rows and in Wan 512. A block is then self-attention with rotary positions, cross-attention to the prompt, and a feed-forward network of two matrices, the whole modulated by the timestep as in that chapter (adaLN, adaptive layer normalisation: small networks turn the timestep into a scale, a shift and a gate for each normalisation and each sublayer). The FLOP count of such a block, with $d$ the width and $d_{\mathrm{ff}}$ the feed-forward width, is the four $d \times d$ matrices of self-attention, the two of the cross-attention that act on the tokens, and the two feed-forward matrices on every token, plus the $4N^2 d$ of the scores and the weighted sum of Chapter 7 of Volume V:

$$
N\,(12\,d^2 + 4\,d\,d_{\mathrm{ff}}) \;+\; 4\,N^2 d \;+\; \bigl(4\,N_{\mathrm{ctx}}\,d^2 + 4\,N\,N_{\mathrm{ctx}}\,d\bigr),
$$

the last bracket being the projection of the prompt's rows to keys and values and the cross-attention scores. The first term grows with $N$ and the second with its square, and they are equal at $N^\star = 3d + d_{\mathrm{ff}}$, below which the matrices dominate a block and above which attention does. The table does the arithmetic for the models of the chapter.

<!-- worked: video_generation.attention_split -->
FLOPs of one block (a multiply-add counted as two) in units of 10^12, from the four d x d matrices of self-attention, the q and o matrices of text cross-attention on the T tokens, a feed-forward of two matrices of width d by f (LTX: d = 4096, f = 4d; Wan 1.3B: d = 1536, f = 8960; Wan 14B: d = 5120, f = 13824), and the scores plus weighted sum of self-attention (4 T^2 d); the text side (k and v matrices on N = 1024 or 512 context rows, 4 N d^2, and the cross-attention scores 4 T N d) is shown separately. Left out: norms, rotation, gates, the embedding and output matrices, and the conditioning networks (each a small fraction of the block matrices). 'Attention share' is the self-attention scores over the whole block. 'Forward' multiplies the block by its depth (48, 30, 40). 'Crossover' is the T at which self-attention scores equal the token-side linear layers, T = c / (4 d) with c the per-token linear FLOPs.

| clip | tokens T | linear / block | self-attention / block | text side / block | attention share | forward (10^12) | crossover T |
|---|---:|---:|---:|---:|---:|---:|---:|
| LTX-2.5 22B, 768 x 512, 121 frames | 6,144 | 2.89 | 0.62 | 0.172 | 0.17 | 176.5 | 28,672 |
| LTX-2.5 22B, 1536 x 1024, 121 frames | 24,576 | 11.54 | 9.90 | 0.481 | 0.45 | 1052.2 | 28,672 |
| Wan 1.3B, 832 x 480, 81 frames | 32,760 | 2.73 | 6.59 | 0.108 | 0.70 | 283.0 | 13,568 |
| Wan 14B, 1280 x 720, 81 frames | 75,600 | 45.19 | 117.05 | 0.846 | 0.72 | 6523.3 | 29,184 |
<!-- /worked -->

The LTX clip of 6144 tokens is below its crossover of 28 672, and attention is 17 percent of a block. A clip at 1536 by 1024, 24 576 tokens, is at 45 percent, and Wan's default clip, 32 760 tokens against a crossover of 13 568, is already at 70. This is the arithmetic behind the opening of `crates/wan/src/lib.rs`: a materialised score matrix for one head at 32 760 tokens is $32\,760^2 \times 4$ bytes, 4.3 GB, and at the 14B model's 75 600 tokens it is 22.9 GB, so attention that never builds the matrix, the fused kernels of Volume V's [chapter on flash attention](../v05-sequences/13-flash-attention.md), is a requirement for running Wan at all. A comment in `crates/ltxv/src/block.rs` gives the matching figure for LTX, with one slip: it calls $32 \times 3520^2 \times 4$ bytes "1.59 GiB", and that is 1.59 gigabytes, 1.48 gibibytes.

## Where a token is in time, height and width

Attention that ignores order cannot tell the first frame from the last, so every token needs a position, and a video token needs three. Volume V's chapter on positions introduced the multi-axis rotary embedding for exactly this: the channel pairs of a head are shared among the axes, and a pair turns by the position on its own axis times its own rate. The two models make different choices inside that frame, and the differences are about units.

Wan keeps to the integer grid. A head of 128 channels has 64 pairs, divided as $[22, 21, 21]$, the frame axis taking the remainder (`rope_axes_dims` in `crates/wan/src/config.rs`). Pair $j$ of an axis of width $d$ turns by (the integer position on the axis) times $10000^{-2j/d}$, with $d$ that axis's own width, so the three axes have three ladders.

<!-- worked: video_generation.wan_rope_table -->
Wan 1.3B has 12 heads of width 128, so 64 rotation pairs per head, divided as [c - 2 (c // 3), c // 3, c // 3] with c = 64: 22 pairs on the frame axis and 21 on each of the other two. Pair j of an axis of width d turns by (the integer position on that axis) * 10000^(-2 j / d). The last column is how many positions the default 832 x 480, 81-frame clip uses on the axis (21 latent frames, 480 / 16 = 30 rows, 832 / 16 = 52 columns of patches).

| axis | pairs | width d | fastest rate | slowest rate | slowest wavelength | positions used |
|---|---:|---:|---:|---:|---:|---:|
| frame | 22 | 44 | 1 | 0.000152 | 41,339 | 21 |
| height | 21 | 42 | 1 | 0.000155 | 40,523 | 30 |
| width | 21 | 42 | 1 | 0.000155 | 40,523 | 52 |
<!-- /worked -->

The slowest wavelength of every axis, about 40 000 positions, is far beyond the 1024 positions of the tables (`MAX_SEQ_LEN` in `crates/wan/src/rope.rs`), so no pair wraps, and the clip's 21 frames, 30 rows and 52 columns are small corners of them.

LTX uses physical units. A token's position on an axis is the midpoint of the span it covers, in seconds on the frame axis and in pixels on the other two, divided by the axis's maximum (20 seconds, 2048 pixels) and mapped to $[-1, 1]$:

$$
\bar p_a = \frac{2\,\mathrm{mid}_a}{p_{\max,a}} - 1, \qquad
\theta_{k,a} = r_k\,\bar p_a, \qquad
r_k = \mathrm{base}^{\,k/(n_{\mathrm{band}} - 1)}\,\frac{\pi}{2}, \quad \mathrm{base} = 10000 .
$$

The number of bands is $n_{\mathrm{band}} = \lfloor d/6 \rfloor$, 682 for $d = 4096$, and every band has one rate $r_k$ that its three axes share. The frame axis is where the causal grid of the last chapter enters: latent frame $k$ covers the pixel frames $\max(0, 8k - 7)$ to $\max(0, 8k + 1)$, divided by the frame rate, so the first latent frame spans one pixel frame and each later one eight (`real_pixel_positions` in `crates/ltxv/src/pipeline.rs`).

<!-- worked: video_generation.ltx_position_table -->
The frame axis of the first four latent frames, in seconds: latent frame k spans the pixel frames max(0, 8k - 7) .. max(0, 8k + 1) (half-open), divided by the frame rate. The position that enters the rotation is the midpoint divided by the axis maximum (20 seconds), mapped to [-1, 1] by 2 mid / 20 - 1.

| fps | k | start (s) | end (s) | midpoint (s) | mid / 20 | 2 mid / 20 - 1 |
|---:|---:|---:|---:|---:|---:|---:|
| 24 | 0 | 0.0000 | 0.0417 | 0.0208 | 0.001042 | -0.997917 |
| 24 | 1 | 0.0417 | 0.3750 | 0.2083 | 0.010417 | -0.979167 |
| 24 | 2 | 0.3750 | 0.7083 | 0.5417 | 0.027083 | -0.945833 |
| 24 | 3 | 0.7083 | 1.0417 | 0.8750 | 0.043750 | -0.912500 |
| 12 | 0 | 0.0000 | 0.0833 | 0.0417 | 0.002083 | -0.995833 |
| 12 | 1 | 0.0833 | 0.7500 | 0.4167 | 0.020833 | -0.958333 |
| 12 | 2 | 0.7500 | 1.4167 | 1.0833 | 0.054167 | -0.891667 |
| 12 | 3 | 1.4167 | 2.0833 | 1.7500 | 0.087500 | -0.825000 |

The height axis of latent rows 0, 1, 2 (32 pixels each, maximum 2048 pixels); the width axis is the same:

| row | pixels | midpoint | mid / 2048 | 2 mid / 2048 - 1 |
|---:|---|---:|---:|---:|
| 0 | 0 .. 32 | 16 | 0.007812 | -0.984375 |
| 1 | 32 .. 64 | 48 | 0.023438 | -0.953125 |
| 2 | 64 .. 96 | 80 | 0.039062 | -0.921875 |
<!-- /worked -->

The two halves of the first table differ in one number, the frame rate, and the frame axis of the same latent grid sits at twice the seconds at 12 frames per second that it does at 24. A model that is told its clip runs at 12 and one told 24 see the same tokens in different places, which is the right thing for a model whose motion is meant to be in seconds.

The rates $r_k$ are slow for low bands and fast for high ones, as the ladder of Volume V's chapter was, and the table shows how much.

<!-- worked: video_generation.ltx_angle_table -->
inner_dim = 4096 gives L = 4096 // (2 * 3) = 682 bands, 2046 angles per token and 2 identity columns in front of a table of 2048 columns. Band k has rate indices[k] = 10000^(k / 681) * pi / 2, and the angle of a token on an axis is that rate times 2 mid / max - 1. Latent frames 1 and 2 of a 24 frames per second clip have the positions -0.979167 and -0.945833 on the frame axis.

| band k | rate | angle, latent frame 1 | angle, latent frame 2 | turn between them |
|---:|---:|---:|---:|---:|
| 0 | 1.5708 | -1.5381 | -1.4857 | 0.0524 |
| 1 | 1.5922 | -1.5590 | -1.5059 | 0.0531 |
| 340 | 156.0210 | -152.7705 | -147.5698 | 5.2007 |
| 681 | 15707.9633 | -15380.7140 | -14857.1153 | 523.5988 |
<!-- /worked -->

Band 0 sweeps $\pi$ radians over the whole 20 seconds of the frame axis and moves 0.05 radian between two neighbouring latent frames; band 681 moves 523.6 radians between them, 83 whole turns. Each head sees only a window of the ladder. The 4096 channels of a block are 32 heads of 128, the rotation pairs channel $j$ of a head with channel $j + 64$ (the split pairing of Volume V), and the per-token table of 2048 columns is built once, with two identity columns in front, and cut into 32 consecutive chunks of 64 (`ltx_rope_tables` in `crates/ltxv/src/rope.rs`). Head 0 therefore carries the slowest bands and head 31 the fastest, where a rotary embedding that builds one table shared by every head, as `crates/dit/src/rope.rs` does, gives each head the whole ladder; the module comment of `crates/ltxv/src/rope.rs` gives this difference as the reason it cannot reuse that code.

<!-- worked: video_generation.ltx_head_table -->
The table of 2048 columns is cut into 32 consecutive chunks of 64 columns, one per head; column c >= 2 holds band (c - 2) // 3 and axis (c - 2) mod 3, with the axes in the order frame, height, width.

| head | columns | first column holds | last column holds |
|---:|---|---|---|
| 0 | 0 .. 63 | identity | band 20, axis 1 |
| 1 | 64 .. 127 | band 20, axis 2 | band 41, axis 2 |
| 2 | 128 .. 191 | band 42, axis 0 | band 63, axis 0 |
| 30 | 1920 .. 1983 | band 639, axis 1 | band 660, axis 1 |
| 31 | 1984 .. 2047 | band 660, axis 2 | band 681, axis 2 |
<!-- /worked -->

Because the unit is built into the position, getting it wrong is silent. Commit `091c3f9ef` (20 August 2026) replaced positions built as raw latent-grid integers by the real ones, and its message and knowledge #48 tell the story: before it, a real-weight generation produced fine-grained textured noise at every shape and every sampler setting tried, with finite values and no crash, because the model ran its trained frequency table over the wrong slice of itself. Latent row 3 is $[3, 4)$ in one convention and $[96, 128)$ in the other, the same formula fed a value 32 times off. Every other part of the pipeline had been proved against the reference, and the real-weight parity tests did not catch it because their goldens had been built from the same raw positions on both sides. The lesson is the one the last chapter drew about the patch order: a gate that is built with the simplification it should check proves the sequence of operations and never the units.

One more consequence of the normaliser is worth reading off the table. A window of the long-form plan below has at most 34 latent frames, 265 pixel frames, 11.04 seconds at 24 frames per second, so its frame positions stay inside the $[-1, 1]$ range of the 20-second normaliser, and each window is re-based to its own time origin.

## A schedule that depends on the clip

Volume VIII's chapter on flow matching derived the timestep shift from redundancy: if a data point is many copies of one value, each with its own noise, the signal-to-noise ratio of the whole is higher than that of one copy, and a grid of steps that suits one copy wastes its steps on the clean end. A clip is the extreme case, since neighbouring tokens in space and in time are nearly copies. The scheduler of LTX therefore makes the shift a function of the token count. Its strength is the straight line through $\mu = 0.95$ at 1024 tokens and $\mu = 2.05$ at 4096, continued past both ends without a clamp,

$$
\mu(N) = 0.95 + \frac{N - 1024}{3072}\,(2.05 - 0.95),
$$

with $k = e^{\mu}$ in the shift $\sigma' = k\sigma/(1 + (k - 1)\sigma)$ of that chapter (`ltx2_sigmas` in `crates/diffusion/src/scheduler.rs`). The sampler's noise levels are a uniform ramp from 1 to 0, shifted, and then stretched so that the last non-zero level lands on a terminal value, 0.1: with $q$ the last non-zero shifted value, every non-zero value $v$ becomes $1 - (1 - v)(1 - 0.1)/(1 - q)$, which keeps the final step from ending too close to a fully clean latent.

<!-- worked: video_generation.schedule_shift_table -->
mu(T) is the straight line through (1024, 0.95) and (4096, 2.05), continued without a clamp. A 4-step schedule: s_i = 1 - i/4, shifted to e^mu / (e^mu + 1/s - 1), then stretched: with q the last non-zero shifted value, every non-zero value v becomes 1 - (1 - v) (1 - 0.1) / (1 - q), so the last non-zero value is exactly 0.1. The first value is 1 and the last is 0 in every row.

| clip | tokens T | mu | e^mu | sigma 1 | sigma 2 | sigma 3 |
|---|---:|---:|---:|---:|---:|---:|
| one 768 x 512 latent frame | 384 | 0.7208 | 2.056 | 0.7884 | 0.5037 | 0.1000 |
| 1024 tokens | 1,024 | 0.9500 | 2.586 | 0.8086 | 0.5327 | 0.1000 |
| 2048 tokens | 2,048 | 1.3167 | 3.731 | 0.8344 | 0.5732 | 0.1000 |
| a 1024 x 1024 picture | 4,096 | 2.0500 | 7.768 | 0.8671 | 0.6316 | 0.1000 |
| 768 x 512, 121 frames | 6,144 | 2.7833 | 16.173 | 0.8838 | 0.6651 | 0.1000 |
| 1536 x 1024, 121 frames | 24,576 | 9.3833 | 11888.577 | 0.9000 | 0.6999 | 0.1000 |
<!-- /worked -->

Reading down the table, a bigger clip is pushed harder toward noise, as the derivation says. The row of 4096 tokens has the same $\mu$ of 2.05 that a 1024 by 1024 picture would, and the shift of FLUX.2 in Volume VIII's chapter is of the same order there (2.0 at 50 steps and 1.15 at 200). The stretch has a second effect that the table makes visible: as $N$ grows the unstretched levels crowd toward 1 and the stretch undoes it, so that for a four-step schedule the levels tend to 1, 0.9, 0.7, 0.1 and stop moving. The 24 576-token row, with an extrapolated $\mu$ of 9.4, has already arrived.

None of this is used to generate with the real checkpoint. `generate` in `crates/ltxv/src/pipeline.rs` takes the shifted ramp only for the random-weight smoke configuration, and with `ltx25_22b` it takes a constant table, `LTX2_DISTILLED_SIGMAS`, eight steps long, copied from the reference and compared with it bit for bit. The code's comment says why: the distilled checkpoint was trained to denoise from a handful of specific levels, and a generic ramp of plausible numbers produced unrecognisable output (a claim of the comment, which the code acts on by ignoring `--steps` for that checkpoint). The table is not a shifted ramp.

<!-- worked: video_generation.distilled_table -->
The eight-step schedule of the distilled checkpoint is a fixed table (it is not the output of the shifted ramp above), and the second stage of a two-stage run takes its last four entries. log SNR = 2 ln((1 - s) / s).

| step | sigma | log SNR | step size | note |
|---:|---:|---:|---:|---|
| 0 | 1.000000 | -inf | 0.006250 |  |
| 1 | 0.993750 | -10.138 | 0.006250 |  |
| 2 | 0.987500 | -8.739 | 0.006250 |  |
| 3 | 0.981250 | -7.915 | 0.006250 |  |
| 4 | 0.975000 | -7.327 | 0.065625 |  |
| 5 | 0.909375 | -4.612 | 0.184375 | stage 2 starts here |
| 6 | 0.725000 | -1.939 | 0.303125 |  |
| 7 | 0.421875 | +0.630 | 0.421875 |  |
| 8 | 0.000000 | +inf |  |  |
<!-- /worked -->

Four of the eight steps sit between 1 and 0.975, steps of 0.00625; the other four cover the remaining 0.975 in steps that grow from 0.066 to 0.42. In log signal-to-noise the first four hops end at $-7.3$, and the last non-zero level is at $+0.63$.

The table is meant for a particular token count. The reference runs it at half the requested width and height, a quarter of the tokens, carries the latent up with a learned spatial upscaler that doubles the grid, and refines it with the last three entries of the table, $0.909375, 0.725, 0.421875$ and then 0, at full size. Commit `d679b3eae` found that brain had been running the table in one stage at the requested size, and that it fails above a token count: the ratio of the opening was 1.04 at 5600 tokens and 14.66 at 8160. `SINGLE_STAGE_MAX_TOKENS` is 6144, between the largest count measured good and the smallest measured bad, and `should_two_stage` chooses two stages above it when both sides are multiples of 64, so that halving lands on the 32-pixel grid. The default clip of this chapter, 6144 tokens, sits exactly on the ceiling and runs in one stage. The code says honestly that where in the interval the cliff really lies has not been measured; the constant has only to separate the two. The same commit found a second defect in the upscaler path, which had skipped the per-channel un-normalise and re-normalise the upscaler expects, and which cost half the latent's variance; the old parity test missed it because cosine similarity cannot see a scale, the lesson of knowledge #2. The two-stage plan is also cheaper than it looks, in work.

<!-- worked: video_generation.two_stage_work -->
The video-stream forward of the table above, 1536 x 1024 output, 121 frames: stage 1 runs 8 forwards on the half-size latent and stage 2 three on the full latent after a spatial upscale; the comparison is eight forwards at full size.

| plan | forwards x tokens | FLOPs (10^15) |
|---|---:|---:|
| two stages | 8 x 6,144 + 3 x 24,576 | 4.57 |
| one stage at full size | 8 x 24,576 | 8.42 |

The two-stage plan does 1.84 times less arithmetic in the denoiser than the single stage at full size.
<!-- /worked -->

## Tokens that are already finished

So far the clip has been all noise. Conditioning means some of it is not: a first frame given by the user, a last frame, a still in the middle, the tail of the clip that came before, or, for sound, a soundtrack. The mechanism has three parts that have to agree: the sampler pins the held token's estimate to its content, the network is told that the token is held, and the conversion from velocity to clean estimate runs at the token's own timestep. The history of the crate is the history of getting the three right one at a time.

Each token $n$ has a mask $m_n \in [0, 1]$, 1 for a token that is generated and 0 for one that is held. A held token starts as its clean content, and at strength $s$ it starts as the mixture $(1 - m_n)\,\mathbf{x}^{\mathrm{c}}_n + m_n\,\boldsymbol{\varepsilon}_n$ with $m_n = 1 - s$. At every step the network is told the token's own timestep $t_n = m_n \sigma$, not the schedule's $\sigma$, and the clean estimate that the loop forms is $\hat{\mathbf{x}}_0 = \mathbf{x} - t_n\,\mathbf{u}_\theta$, the conversion of Volume VIII with the token's timestep in place of the scalar. The estimate is then pulled back toward the clean content, $m_n \hat{\mathbf{x}}_0 + (1 - m_n)\,\mathbf{x}^{\mathrm{c}}_n$, before the sampler steps. In the book's pseudocode, for the deterministic sampler:

```text
function denoise_clip(x: array<real>[N, C], mask: array<real>[N], clean: array<real>[N, C],
                      sigmas: list<real>,
                      velocity: function(array<real>[N, C], array<real>[N]) -> array<real>[N, C]) -> array<real>[N, C]:
    require len(sigmas) >= 2 and sigmas[0] > 0.0 and sigmas[len(sigmas) - 1] == 0.0
    y = zeros(real, [N, C])
    for n in 0 .. N:
        y[n, :] = (1.0 - mask[n]) * clean[n, :] + mask[n] * x[n, :]
    for i in 0 .. len(sigmas) - 1:
        t = zeros(real, [N])
        for n in 0 .. N:
            t[n] = mask[n] * sigmas[i]
        v = velocity(y, t)
        for n in 0 .. N:
            x0 = y[n, :] - t[n] * v[n, :]
            x0 = mask[n] * x0 + (1.0 - mask[n]) * clean[n, :]
            if sigmas[i + 1] == 0.0:
                y[n, :] = x0
            else:
                r = sigmas[i + 1] / sigmas[i]
                y[n, :] = r * y[n, :] + (1.0 - r) * x0
    return y
```

The step is Volume VIII's Euler step written in terms of the clean estimate, $\mathbf{x} \leftarrow (\sigma_{i+1}/\sigma_i)\,\mathbf{x} + (1 - \sigma_{i+1}/\sigma_i)\,\hat{\mathbf{x}}_0$, which is `euler_ancestral_step` of `crates/diffusion/src/scheduler.rs` at its noise strength `eta` equal to zero (`eta` scales the fresh noise the ancestral step adds after stepping: 0 gives the deterministic loop above, 1 the full ancestral step). The ancestral sampler that the distilled checkpoint runs adds fresh noise after the step and then applies the blend to the stepped latent a second time; the comment on `Frozen` in the pipeline explains why neither application can be dropped for a token whose mask is strictly between 0 and 1. A network that returns a velocity of 1 for every token makes the loop checkable by hand, and the table follows three tokens through the eight steps of the distilled table.

<!-- worked: video_generation.frozen_loop -->
Three tokens of one channel through the loop of the pseudocode on the eight-step distilled schedule, with a network that returns the velocity 1 for every token whatever it is given. The free token has mask 1 and starts at 2. The half-held token has mask 0.5, clean content 10 and starts from the noise draw -2 (the loop first replaces it by 0.5 * 10 + 0.5 * (-2) = 4). The pinned token has mask 0, clean content 5 and starts from the draw 7 (the loop replaces it by 5). Row i is the state before step i; the last row is the result.

| step | free | half held | pinned |
|---:|---:|---:|---:|
| 0 | 2.000000 | 4.000000 | 5.000000 |
| 1 | 1.993750 | 4.017188 | 5.000000 |
| 2 | 1.987500 | 4.034439 | 5.000000 |
| 3 | 1.981250 | 4.051755 | 5.000000 |
| 4 | 1.975000 | 4.069136 | 5.000000 |
| 5 | 1.909375 | 4.252326 | 5.000000 |
| 6 | 1.725000 | 4.788900 | 5.000000 |
| 7 | 1.421875 | 5.802508 | 5.000000 |
| 8 | 1.000000 | 7.795785 | 5.000000 |

The free token loses exactly the sum of the step sizes, 1, and ends at 1; the pinned token never moves.
<!-- /worked -->

A free token loses exactly the sum of the step sizes, 1, and a pinned token does not move at all. The half-held token starts from the mixture 4 and ends at 7.795785, pulled up toward its clean content 10 by the blend and down by the network's estimate, which was told its timestep was half the schedule's.

The first two tables of the mechanism, for one token of clean content 10 whose noise draw is $-2$, show what the strength does to the start and to the timestep.

<!-- worked: video_generation.conditioning_table -->
A token whose clean content is c = 10, with initial noise draw n = -2, conditioned at strength s: the mask is m = 1 - s, the token starts at (1 - m) c + m n and is announced to the network at the timestep m * sigma. The last two columns are that timestep at the first sigma (1) and at the last non-zero sigma of the distilled table (0.421875).

| strength s | mask m | start value | timestep at sigma = 1 | timestep at sigma = 0.421875 |
|---:|---:|---:|---:|---:|
| 1 | 0 | +10.00 | 0 | 0.000000 |
| 0.8 | 0.2 | +7.60 | 0.2 | 0.084375 |
| 0.5 | 0.5 | +4.00 | 0.5 | 0.210938 |
| 0 | 1 | -2.00 | 1 | 0.421875 |
<!-- /worked -->

The first part came with the first conditioning code, commit `091c3f9ef`, the commit of the positions above, which introduced `Frozen` and `post_process_latent` for a start and an end frame. The other two were each missing for a while, and each time the clip came out wrong in a way that pointed away from the cause. Commit `bf00210e5` and knowledge #49 are the second part. The loop had passed the schedule's $\sigma$ to the network for every token, and with a frozen first frame the result was a conditioned frame reproduced perfectly and every other frame a horizontal-line mesh. The sampler's blend pinned the held frame no matter what the network was told, which is why it decoded cleanly and why the symptom read as a capacity limit. The network builds one scale, shift and gate row per token out of the timestep, so a token of clean content labelled as pure noise was modulated as noise and dragged every genuinely noisy token with it through attention. The rows are cheap: `distinct_rows` in `crates/ltxv/src/dit.rs` computes the timestep network once per distinct timestep, which for a clip with one pinned frame is two. The lesson of the entry is the one to keep: conditioning has two halves, what the sampler does with the output and what the model is told about the input, and a held output tells you nothing about whether the model understood.

Commit `739ad8fa1` (21 August 2026) is the third part. Every clip conditioned on a first frame had a colour defect: frame 0 oversaturated, the frames after it washed out, then the colour climbed back for the second half. The clean-estimate conversion ran at the schedule's scalar $\sigma$, so at the terminal step of the distilled table, 0.421875, a pinned token's estimate was moved by a full $0.421875\,\mathbf{u}_\theta$ instead of by nothing, and since the ancestral loop returns that estimate without re-pinning on its last step, the last thing to touch the anchor was the thing that corrupted it. The test that pins it was red at $4.578125 = 5 - 0.421875$ before the fix.

<!-- worked: video_generation.clean_estimate_table -->
One token at the terminal step of the distilled schedule, sigma = 0.421875, holding x = 5 with clean content c = 5, and a network that returns the velocity v = 1. The clean estimate is x - v t, with t the timestep: the scalar sigma for every token ('scalar'), or m sigma for a token with mask m ('per token'). The last column blends the per-token estimate with the clean content, m * estimate + (1 - m) * c, which is what the loop keeps.

| strength s | scalar sigma | per-token timestep | after the blend |
|---:|---:|---:|---:|
| 1 | 4.5781250 | 5.0000000 | 5.0000000 |
| 0.5 | 4.5781250 | 4.7890625 | 4.8945312 |
| 0 | 4.5781250 | 4.5781250 | 4.5781250 |
<!-- /worked -->

The scalar column is the defect and the per-token column is the fix; the third column, with the blend, is what the partially held token keeps. The commit's own check is worth copying for any mask: the defect existed only with `eta = 1`, because the deterministic loop re-pins before stepping and is immune, and a gate written with `eta = 0` passed against the bug at bit-identical numbers.

How the held tokens sit in the sequence is the last choice, and it has a price in pairs. A single still at frame 0 and nothing else overwrites the first latent frame of the clip in place: it costs no tokens but freezes a whole latent frame, 384 of 6144 tokens, one sixteenth of a 121-frame clip and half of a 9-frame one. Any other request, with an end still, a still in the middle or the tail of an earlier clip, takes the keyframe path where each still is appended as extra tokens carrying its own positions, so the generated tokens stay free and the still is guidance; `conditioning_block_count` in the pipeline counts how many tokens are appended. A mid-clip still is the exception, pinned into the latent slot whose pixel span contains the frame it names (pixel frame 60 of a 121-frame clip is in latent frame 8, which covers pixel frames 57 to 64) rather than appended.

<!-- worked: video_generation.anchor_cost -->
A 768 x 512, 121-frame clip is 6,144 tokens (16 latent frames of 384). A pinned still overwrites a latent frame's tokens in place and adds nothing; an appended guide adds 384 tokens to the sequence. 'Free' is the number of tokens the model is still generating.

| conditioning | sequence length | self-attention pairs | pairs vs text only | free tokens |
|---|---:|---:|---:|---:|
| text only | 6,144 | 37,748,736 | 1.0000 | 6,144 |
| first frame pinned into its slot (image to video) | 6,144 | 37,748,736 | 1.0000 | 5,760 |
| a mid-clip still pinned into its slot | 6,144 | 37,748,736 | 1.0000 | 5,760 |
| first still appended as a guide | 6,528 | 42,614,784 | 1.1289 | 6,144 |
| first and last stills appended | 6,912 | 47,775,744 | 1.2656 | 6,144 |
| first, middle and last stills, middle pinned | 6,912 | 47,775,744 | 1.2656 | 5,760 |
<!-- /worked -->

A first frame appended as a guide raises the self-attention pairs by 12.9 percent, a first and a last still appended by 26.6 percent, and a still pinned into its slot adds nothing. A different conditioning route is the one Wan takes. Its image-to-video checkpoint widens the input from 16 channels to 36, 16 of latent, 4 of mask and 16 of the encoded first frame, so the held frame enters as channels and not as tokens with a timestep, and a CLIP vision tower adds 257 tokens through a separate pair of key and value matrices in every block (`Task::I2v` in `crates/wan/src/config.rs`). It is a route brain knows the shape of and does not run: the importer refuses an image-to-video checkpoint by name, because the 36-channel pipeline and the vision tower do not exist in the crate. The extra weights of that checkpoint, counted from the manifest, are 2.1 billion on top of the 14B text-to-video model.

## Sound as a second stream

LTX-2.5 generates sound with the picture, and not by running a second model after the first. **Joint audio-video generation** is one transformer that denoises a video latent and an audio latent together, each in its own stream with its own widths, coupled in every block by cross-attention in both directions, so that the sound can depend on the picture and the picture on the sound at every step. The audio latent of a clip is derived from the clip's duration. At 25 audio tokens a second (16 000 samples, a hop of 160 samples, 4 mel frames per latent frame), a clip of $F$ frames at $\rho$ frames per second has $N_{\mathrm a} = \mathrm{round}(25F/\rho)$ tokens, each a 128-wide vector, the 8 channels by 16 mel bins of the latent (`crates/ltxv/src/audio.rs`). The 121-frame clip at 24 frames per second lasts 5.0417 seconds and has 126.

The tokens of the two streams meet in time, which is why their positions are in seconds. An audio token $i$ spans $\max(0, 4i - 3)/100$ to $\max(0, 4(i + 1) - 3)/100$ seconds, the causal offset of the audio autoencoder, and every cross-attention between the streams uses one rotary table over time alone, normalised by 20 seconds, built from the video's frame axis and the audio's single axis (`latent_bounds` in `audio.rs`, and `ltx_rope_tables` run at one axis). A video frame and an audio token at the same moment attend to each other because they carry the same coordinate, with no resampling in between.

<!-- worked: video_generation.audio_grid_table -->
A clip of 25 frames at 24 frames per second lasts 1.0417 seconds and has ta = round(25/24 * 25) = 26 audio tokens. Audio token i spans max(0, 4 i - 3) / 100 to max(0, 4 (i + 1) - 3) / 100 seconds (token 0 spans 0 .. 0.01, token 1 spans 0.01 .. 0.05, token 2 spans 0.05 .. 0.09, and so on). The table counts the audio tokens whose midpoint lies in each latent frame's span.

| latent frame | span (s) | audio tokens | which |
|---:|---:|---:|---:|
| 0 | 0.0000 .. 0.0417 | 2 | 0 .. 1 |
| 1 | 0.0417 .. 0.3750 | 8 | 2 .. 9 |
| 2 | 0.3750 .. 0.7083 | 8 | 10 .. 17 |
| 3 | 0.7083 .. 1.0417 | 8 | 18 .. 25 |

The sound decodes to (4 * 26 - 3) = 101 mel frames of 160 samples, 16160 samples, 1.0100 seconds at 16000 samples a second: 0.0317 seconds shorter than the picture.
<!-- /worked -->

The sound is three mel frames shorter than the picture, 30 thousandths of a second at the exact token count, because the first latent frame of the causal audio autoencoder covers one mel frame instead of four. That is the reference's arithmetic, and `pad_to_seconds` extends the end by repeating the last sample so that a container can hold two streams of equal length.

![One block of the two-stream transformer drawn as two rows of boxes: video tokens above and audio tokens below, each with self-attention using rotary positions, a cross-attention to the prompt, and a feed-forward network, and between the two rows one tall box for the cross-attention from audio to video and from video to audio.](img/av-block.svg)

A block of the real model runs, in order, the video stream's self-attention and prompt cross-attention, the audio stream's own two, the cross-attention in both directions on a snapshot of each stream taken after those, and then the two feed-forward networks (`LtxAvBlock` in `crates/ltxv/src/block.rs`). Both cross-attentions run at the audio stream's width of 2048 and its 32 heads of 64, and each is gated by a single row taken from a timestep built from the other stream's scalar noise level. A stream that is entirely held has noise level 0 for that purpose (`stream_sigma`), which is how a given soundtrack works: audio-to-video encodes the soundtrack through the audio autoencoder and holds it clean at timestep 0, the same mechanism as a held first frame, and the video is denoised against it. Guidance reaches across the streams too: an isolated-modality pass cuts both directions of cross-attention, and the combination is on the clean estimate, with $\hat{\mathbf{x}}^{\mathrm{cond}}$ the clean estimate of the conditional pass and $\hat{\mathbf{x}}^{\mathrm{neg}}$, $\hat{\mathbf{x}}^{\mathrm{stg}}$, $\hat{\mathbf{x}}^{\mathrm{iso}}$ those of the others,

$$
\hat{\mathbf{x}} = \hat{\mathbf{x}}^{\mathrm{cond}} + (g - 1)(\hat{\mathbf{x}}^{\mathrm{cond}} - \hat{\mathbf{x}}^{\mathrm{neg}}) + g_{\mathrm{s}}\,(\hat{\mathbf{x}}^{\mathrm{cond}} - \hat{\mathbf{x}}^{\mathrm{stg}}) + (g_{\mathrm{m}} - 1)(\hat{\mathbf{x}}^{\mathrm{cond}} - \hat{\mathbf{x}}^{\mathrm{iso}}),
$$

with the negative-prompt scale $g$, the scale $g_{\mathrm{s}}$ of the pass that bypasses self-attention in chosen blocks (spatiotemporal guidance), and the scale $g_{\mathrm m}$ of the isolated-modality pass (`GuiderParams::combine` in `crates/ltxv/src/guidance.rs`). A term whose scale makes it vanish is not run, so the number of forwards in a step is one plus the number of switched-on terms.

What the second stream costs is small in arithmetic and larger in weights.

<!-- worked: video_generation.av_cross_cost -->
The 121-frame, 768 x 512 clip at 24 frames per second has T = 6,144 video tokens and ta = 126 audio tokens (width 2048, 32 heads of 64). The video stream's self-attention scores 37,748,736 pairs per head; each cross-attention direction scores T * ta = 774,144, 0.021 of that. FLOPs of one block (query, key, value and output matrices, scores and weighted sum; norms, gates and the adaLN tables left out). The shapes are these. Video stream: the block of the attention-split table (width 4096, feed-forward width 4 * 4096, 1024 context rows of width 4096). Audio stream, width 2048: self-attention with four 2048 x 2048 matrices over the 126 audio tokens and the scores 4 ta^2 2048; text cross-attention whose q and o matrices act on the 126 audio tokens and whose k and v matrices act on the 1024 context rows, taken at width 2048 like the video's (4 * 1024 * 2048^2), with scores 4 ta 1024 2048; a feed-forward of two 2048 x 4 * 2048 = 2048 x 8192 matrices. Cross-attention in each direction works at width 2048, the audio width: with the query stream on the left, its q matrix maps the query stream's width to 2048, its k and v matrices map the other stream's width to 2048, its o matrix maps 2048 back to the query stream's width, and the scores and weighted sum cost 4 T ta 2048. The columns below therefore count 2 T 4096 2048 (video q) + 2 (2 ta 2048^2) (audio k, v) + 2 T 2048 4096 (video o) + 4 T ta 2048 for audio to video, and 2 ta 2048^2 (audio q) + 2 (2 T 4096 2048) (video k, v) + 2 ta 2048^2 (audio o) + 4 T ta 2048 for video to audio:

| part of the block | FLOPs |
|---|---:|
| video stream | 3,676,492,005,376 |
| audio stream (self, text, feed-forward) | 33,164,394,496 |
| audio to video cross-attention | 214,614,147,072 |
| video to audio cross-attention | 214,614,147,072 |

The audio stream and both cross-attentions add 0.126 of the video stream's FLOPs.
<!-- /worked -->

The cross-attentions score 2 percent of the pairs of the video's self-attention, and the audio stream with both of them adds 12.6 percent to the block's arithmetic. In weights the extra is 27 percent of the transformer, 15.4 percent for the audio stream and 11.6 for the cross-attentions, the counts of the last section. The doc comment on `GenOpts::audio` in the pipeline says two thirds of the checkpoint's tensors are the audio half; that holds for the count of tensors, 56 of the 84 in a block counting the cross-attentions, and not for the count of parameters.

## The prompt, and what Gemma-4 is for

The prompt reaches the transformer as rows, and for LTX the rows come from Gemma-4, a 12-billion-parameter model of which the video generator uses only the text tower (`crates/gemma4`). It is a decoder of 48 layers at width 3840, with 16 query heads, in which five layers in six are sliding-window layers of 8 key-value heads of width 256 and a window of 1024, and every sixth layer is a full-attention layer with a single key-value head of width 512 whose value is its key (`attention_k_eq_v`), so that those eight layers have no value matrix at all. The crate ports the text path only: the vision tower and the audio projector of the unified checkpoint are classified as out of scope by the importer and never read.

What LTX takes from the tower is not its output but all 49 of its hidden states, the embedding output and each layer, and what turns them into the prompt is a projection built for it. For each token, each of the 49 states is scaled to unit root mean square on its own, with no learned gain; the states are interleaved, so that input column $49d + k$ holds coordinate $d$ of state $k$ and not the other way round; the result is multiplied by $\sqrt{d_{\mathrm{out}}/3840}$; and one linear layer takes the 188 160 columns to 4096 for the video stream and, from the same states, to 2048 for the audio (`AggregateEmbed` in `crates/gemma4/src/model.rs`). The rows are then right-padded to 1024, the width of the reference's tokenizer, and an embeddings connector, a small transformer with 128 learnable registers, writes its registers over the padded positions before the blocks read them; each branch carries a mask of its own real tokens.

Commit `7d86e4a79` (21 August 2026) is the story of the projection. It had been written as a plain concatenate-and-project, on the stated premise that the module's structure could not be derived from a tensor header. The premise was false, since the reference module was vendored in the repository, and the guess differed in all three ways above. The commit says the result was a context whose every row sat at a cosine of 0.996 of every other row, a constant that carried no caption, and the code comment adds that two different prompts produced context matrices whose mean rows were at 0.9998 of each other and that generation reproduced one scene whatever it was asked. These are the claims of the commit and a comment, and the code is consistent with them: the 49 raw states differ in magnitude by orders of magnitude, so a flatten without the per-state normalisation is dominated by the largest states' near-constant component, and the wrong column order permutes the weight matrix. The tests that now pin the fix are weight-free: scaling one state of one token by a positive factor must not change that token's output, and one-hot weight rows pin the column order.

Encoding is work, and the tower's share of it is easy to count.

<!-- worked: video_generation.tower_work -->
FLOPs of encoding a prompt of n tokens (in units of 10^12): the 48 layers' matrices at 2 FLOPs per weight per token (embedding lookups are free; the attention among the prompt's own tokens is left out; the tower has 262,144 embedding rows of width 3840, and the layers' own weights are the tower's parameters minus 262,144 * 3840, so the column is 2 * (those weights) * n; layer shapes in the table of counts), then the aggregate projections 188160 -> 4096 (video) and 188160 -> 2048 (audio) at 2 * 188160 * width per token.

| prompt tokens n | text tower | video head | audio head |
|---:|---:|---:|---:|
| 16 | 0.35 | 0.025 | 0.012 |
| 64 | 1.40 | 0.099 | 0.049 |
| 256 | 5.58 | 0.395 | 0.197 |
| 1024 | 22.32 | 1.578 | 0.789 |

One video-stream forward of the 121-frame clip is 176.5 x 10^12 FLOPs, and the prompt is encoded once and reused by all of a clip's forwards.
<!-- /worked -->

A prompt of 64 tokens costs 1.5 x 10^12 FLOPs, under 1 percent of one video forward, and it is paid once per distinct prompt, since the encoded rows are cached under a key of the prompt, the negative prompt, the encoder file and its precision (`crates/ltxv/src/text_cache.rs`). The reference's default negative prompt is far longer than a typical prompt, which is why it is encoded only when some stream's guidance scale asks for it. Wan reads its prompt from a different encoder, umT5-XXL: two prompts in one batch, each masked and zero-padded to 512 rows of width 4096, run before anything else and dropped, because the encoder's 22.72 GB in 32-bit numbers, 5.68 billion weights, does not fit the 24 GB card the crate was written on once its activations are added. The prompt then passes through two matrices of the Wan transformer to its own width, the `text_embedding` of the manifest. Gemma-4 has no role in Wan.

## More than one window

A window of 34 latent frames at 768 by 512, 265 pixel frames, is the largest the crate plans: `LONGFORM_MAX_TOKENS` is 13 200 tokens, the largest count with a recorded real run, and the comment derives it from the size of one adaLN table against the largest buffer the card allows. A clip longer than that cannot be one pass, and the first remedy anyone tries is to chain: decode window $n$, take its last frame, encode it, and use it as window $n+1$'s first frame. Commit `b25b4ad6c` (22 August 2026) says why that fails: it is continuous in position and discontinuous in velocity. A frame says where everything is and nothing about what was moving, so the model invents the motion again at every join, and motion changes, stutters or reverses there.

The remedy it implements carries latent frames instead of a picture. The last 8 latent frames of window $n$'s denoised latent, sliced out before anything is decoded, are held clean at the head of window $n+1$ with timestep 0, exactly the mechanism of the last section with the frame count unpinned (`carry_tail` in `crates/ltxv/src/longform.rs`). Eight latent frames are 57 pixel frames, $1 + 8 \times 7$, the number the code's comment takes from the reference's video-extension training configuration and not from a derivation; the comment records the first guess, that the autoencoder's temporal receptive field sets it, and shows it dead, since the decoder of this checkpoint is not causal and its 42 temporal convolutions reach 14 latent frames to each side, lookahead a rolling window can never have. A seam therefore never decodes exactly, and what the carried frames buy is the diffusion model's motion history.

The bookkeeping is where an off-by-one hides. A window with $c$ carried and $u$ new latent frames decodes $1 + 8(c + u - 1)$ pixel frames; a continuation window drops the pixel frames of its context and emits $8u$, and the first window emits $1 + 8(u - 1)$. The "+1" of the $1 + 8k$ rule belongs to the first window alone, which is why disjoint windows cannot do the job: $n$ windows of the form $1 + 8k_i$ sum to $n + 8\sum k_i$, a legal clip length only for $n = 1$. With carried context the sums close exactly. The plan itself puts the whole budget in the first window, because it is the one that builds structure from nothing, and splits what remains over the fewest continuation windows as evenly as the arithmetic allows. In the book's pseudocode:

```text
function window_plan(frames: int, lh: int, lw: int, context: int, max_tokens: int,
                     align: int) -> list<(int, int, int)>:
    require frames >= 1 and (frames - 1) mod 8 == 0 and context >= 1 and align >= 1
    max_lat = max_tokens // (lh * lw)
    k_total = (frames - 1) // 8
    plan = empty list
    if k_total < max_lat:
        append (0, k_total + 1, 0) to plan
        return plan
    require max_lat >= context + align, "the grid leaves no room for a carried context and a new frame"
    head_new = max_lat - ((max_lat - context) mod align)
    cap_new = max_lat - context
    cap_aligned = align * (cap_new // align)
    remaining = k_total - (head_new - 1)
    n_cont = 1
    while (n_cont - 1) * cap_aligned + cap_new < remaining:
        n_cont = n_cont + 1
    lo = align * max(ceil(max(remaining - cap_new, 0) / align), n_cont - 1)
    hi = align * (min((n_cont - 1) * cap_aligned, remaining - 1) // align)
    want = align * ((remaining - remaining // n_cont) // align)
    constrained = min(max(want, lo), max(hi, lo))
    first = 1 + 8 * (head_new - 1)
    append (0, head_new, 0) to plan
    spread = n_cont - 1
    for i in 0 .. spread:
        units = constrained // align
        new = align * (units // spread + (1 if i < units mod spread else 0))
        append (context, new, first) to plan
        first = first + 8 * new
    last = remaining - constrained
    require last >= 1 and last <= cap_new, "no plan advances every window by a multiple of align"
    append (context, last, first) to plan
    return plan
```

Each entry is the carried frames, the new latent frames and the output pixel frame at which the window's first emitted frame goes. With `align = 1` the quantity `constrained` is the number of new frames of all continuation windows but the last, aimed at an even split and kept between what the last window cannot take and what the others can. The default clip of this chapter, 16 by 24 tokens per latent frame, holds $13\,200 // 384 = 34$ latent frames.

<!-- worked: video_generation.window_table -->
The plan for a 768 x 512 clip (16 x 24 = 384 tokens per latent frame) under the ceiling of 13200 tokens per pass, so at most 34 latent frames, with a context of 8 latent frames. 'Decoded' is 1 + 8 (context + new - 1), 'dropped' the pixel frames of the carried context, 'emitted' 1 + 8 (new - 1) for the first window and 8 new for the others.

121 frames (15 groups of eight after the first):

| window | context | new | tokens | decoded | dropped | emitted | first emitted frame |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 16 | 6,144 | 121 | 0 | 121 | 0 |

The emitted frames add to 121.

265 frames (33 groups of eight after the first):

| window | context | new | tokens | decoded | dropped | emitted | first emitted frame |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 34 | 13,056 | 265 | 0 | 265 | 0 |

The emitted frames add to 265.

601 frames (75 groups of eight after the first):

| window | context | new | tokens | decoded | dropped | emitted | first emitted frame |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 34 | 13,056 | 265 | 0 | 265 | 0 |
| 1 | 8 | 21 | 11,136 | 225 | 57 | 168 | 265 |
| 2 | 8 | 21 | 11,136 | 225 | 57 | 168 | 433 |

The emitted frames add to 601.

1201 frames (150 groups of eight after the first):

| window | context | new | tokens | decoded | dropped | emitted | first emitted frame |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 34 | 13,056 | 265 | 0 | 265 | 0 |
| 1 | 8 | 24 | 12,288 | 249 | 57 | 192 | 265 |
| 2 | 8 | 24 | 12,288 | 249 | 57 | 192 | 457 |
| 3 | 8 | 23 | 11,904 | 241 | 57 | 184 | 649 |
| 4 | 8 | 23 | 11,904 | 241 | 57 | 184 | 833 |
| 5 | 8 | 23 | 11,904 | 241 | 57 | 184 | 1017 |

The emitted frames add to 1201.
<!-- /worked -->

![The three windows of a 601-frame clip laid on its 76-frame latent timeline: a first window of 34 new latent frames, a second that begins with 8 carried frames copied from the end of the first and generates 21 new ones, and a third that begins with 8 carried frames from the second and generates 21 new ones.](img/window-plan.svg)

The first window is a pass over 13 056 tokens and the last two are 11 136, and the decoded frames, minus the 57 dropped for each carried context, add to exactly 601. The plan for a clip with sound has one more condition. The audio stream runs at 25 tokens a second and the video at $\rho/8$ latent frames a second, a ratio that is not an integer at every frame rate, so a seam at an arbitrary latent frame lands part-way into an audio token, and a fraction of a token cannot be carried. A window followed by another must advance, by its latent frames minus its context, a whole number of audio tokens: $n$ latent frames are $200\,n/\rho$ tokens, and the smallest $n$ for which that is whole is the quantum $q$, 3 at 24 and 30 frames per second, 1 at 25, 2 at 16.

<!-- worked: video_generation.window_plan_sound_table -->
For a clip with sound a window that is followed by another must advance by a whole number of audio tokens, n latent frames being 200 n / fps tokens (25 tokens a second, 8 frames per latent frame). The smallest n for which 200 n / fps is a whole number is the quantum q; the plan of the same 601-frame clip with every window but the last advancing by a multiple of q is below (new latent frames of each window).

| fps | tokens per latent frame | quantum q | new latent frames per window |
|---:|---:|---:|---|
| 24 | 25 / 3 | 3 | 32 + 21 + 23 |
| 25 | 8 / 1 | 1 | 34 + 21 + 21 |
| 30 | 20 / 3 | 3 | 32 + 21 + 23 |
| 16 | 25 / 2 | 2 | 34 + 20 + 22 |
<!-- /worked -->

The plan that the table gives for 24 frames per second, 32 then 21 then 23 new frames, has a first window that is two frames short of the budget so that its advance, $32 - 8 = 24$ latent frames, is $200$ audio tokens. `audio_plan` re-derives the layout from the finished plan and refuses any seam that is not whole, rather than rounding, and the tests check that every seam shifts both streams by the same time, $\text{pixel frames} \times 25 = \text{audio tokens} \times \rho$, in integers, so that no float decides it.

The measure of a good seam is the velocity the next window adopts, and a toy shows what the length of the context does to it. A dot moves 1.5 cells per latent frame. A window has produced eight latent frames of it with noise of 0.2 on each position. The next window is given the last $c$ frames. With $c = 1$, a still, it sees a position and nothing else and must draw a velocity from its prior; with $c \geq 2$ it adopts the slope of the line that least squares fits through the frames, the estimate of Volume II that one point cannot give.

<!-- worked: video_generation.seam_toy -->
A dot moves 1.5 cells per latent frame. A window has produced eight latent frames, y_j = 1.5 j + 0.2 z_j for j = 0 .. 7. The next window is given the last c of them and has to continue the motion. With c = 1 (a still) it sees a position and nothing else, so it draws a velocity 2.0 z' from its prior; with c >= 2 it adopts the slope of the least-squares line through the c frames. The error of a seam is |adopted velocity - 1.5|. Draws from `video_generation.seam` (seed 0xd1d849348e07b6e5): for each of 2000 trials, in order, z_0 .. z_7 and then z', each one `normal()` of the book's generator; the means add the 2000 errors left to right. The prediction is the mean of |X| for the exact distribution of the adopted velocity minus the truth: X is normal with mean 1.5 and spread 2.0 for c = 1 (the draw, sign aside), and for c >= 2 normal with mean 0 and spread 0.2 sqrt(12 / (c (c^2 - 1))), the spread of a least-squares slope. Eight frames after the seam the dot is off by eight times the velocity error.

| context frames c | mean velocity error | predicted mean | off after 8 frames |
|---:|---:|---:|---:|
| 1 | 2.006 | 2.025 | 16.05 |
| 2 | 0.222 | 0.226 | 1.78 |
| 3 | 0.114 | 0.113 | 0.91 |
| 4 | 0.071 | 0.071 | 0.57 |
| 8 | 0.025 | 0.025 | 0.20 |
<!-- /worked -->

A still leaves the dot 16 cells off eight frames after the seam on average, since it has to guess the velocity from nothing; two frames cut that to under 2, and eight frames, the carried context, to 0.2. The error falls as the spread of a least-squares slope does, $0.2\sqrt{12/(c(c^2 - 1))}$, which is what the predicted column computes exactly. The toy has none of a real model's structure, and its single parameter, the length of the history, is the one quantity the repository takes from the reference and does not derive. A scene boundary is the one place the carry stops on purpose: a request can be a list of scenes with their own prompts, and the context resets there, so that a new scene is free to be a different subject.

## What the models are made of, and what a clip costs

A model is its tensors, and the tensors of both video models are listed in manifests that the importers check in both directions. The counts follow from the shapes in them.

<!-- worked: video_generation.counts_table -->
Parameters counted from the tensor shapes of each model's checkpoint manifest (weights, biases, norm gains and tables; the 4,349 tensors of the LTX-2.5 transformer are 59 outside the blocks, 84 in each of 48 blocks, and 129 in each of two connectors). Per LTX block: video stream 268,791,872, audio stream 67,297,344, cross-attention 50,585,664. Per Wan block: 46,440,704 (1.3B) and 351,394,304 (14B).

| part | parameters | share of the LTX transformer |
|---|---:|---:|
| LTX-2.5 video stream (48 blocks) | 12,902,009,856 | 0.614 |
| LTX-2.5 audio stream (48 blocks) | 3,230,272,512 | 0.154 |
| LTX-2.5 audio-video cross-attention (48 blocks) | 2,428,111,872 | 0.116 |
| LTX-2.5 two connectors, adaLN networks, input and output | 2,443,631,360 | 0.116 |
| LTX-2.5 transformer, all of it | 21,004,025,600 | 1.000 |
| Gemma-4 text tower (embedding table + 48 layers) | 11,907,350,320 |  |
| Gemma-4 aggregate heads (188160 -> 4096 and -> 2048) | 1,156,061,184 |  |
| Wan 1.3B transformer | 1,418,996,800 |  |
| Wan 14B transformer | 14,288,491,584 |  |
| added by an image-to-video 14B checkpoint | 2,106,387,200 |  |

The rows follow from these shapes. A *linear* layer of a inputs and b outputs has a b weights and b biases unless it is called bias-free. An LTX attention with query width q, key-value width kv and inner width i (i = q for self-attention and text cross-attention; i = 2048 for the audio-video attentions) has the gain vectors of its query and key norms (2 i), a gate with 32 logits computed from the query stream (q 32 weights and 32 biases), and four linear layers with biases: q (q to i), k and v (kv to i) and the output (i to q). A video block (268,791,872) holds two such attentions at width 4096 (self-attention, and cross-attention to the 1024 text rows of width 4096), a bias-free feed-forward of two 4096 x 16384 matrices, a 9 x 4096 table and a 2 x 4096 table of modulation values. An audio block holds the same at width 2048 with a feed-forward of two 2048 x 8192 matrices that has biases (one of 8192 and one of 2048), a 9 x 2048 table and a 2 x 2048 table. The audio-video cross-attention of a block is one attention with q = 4096, kv = 2048, i = 2048 and one with q = 2048, kv = 4096, i = 2048, plus 5 x 4096 and 5 x 2048 modulation values. Outside the blocks: the input projections 128 to 4096 and 128 to 2048 and the output projections 4096 to 128 and 2048 to 128 (each with a bias), the output tables of 2 x 4096, 2 x 2048 and one more row of 4096, and eight adaLN networks (a timestep embedding 256 to d to d, each linear with a bias, then d to r d with a bias, where d is the stream's width and r = 9, 2, 4 and 1 rows for the four networks of each stream). Each of the two connectors has 128 registers of its width and eight blocks, each an attention at that width (i = q = kv) and a feed-forward with biases of two matrices of width 4 times the width. A Wan block (d = 1536 and f = 8960 for 1.3B; d = 5120 and f = 13824 for 14B) holds a 6 x d modulation table, the gain and bias of one more norm (2 d), two attentions (self and text cross-attention) of four d x d linear layers with biases and the gains of two norms (2 d), and a feed-forward d to f to d with biases; outside the blocks are a patch embedding (16 channels x 4 pixels of d weights and d biases), a text embedding (4096 to d to d, with biases), a timestep embedding (256 to d to d), a time projection (d to 6 d), and a head (d to 4 x 16 values) with a 2 x d table. An image-to-video checkpoint widens the patch embedding to 36 channels, adds k and v layers with biases and one norm gain for the image tokens in every block, and an image projection (a norm of width 1280 with gain and bias, 1280 to 1280 and 1280 to d with biases, and a final norm of width d with gain and bias). Gemma-4's text tower has 262,144 embedding rows of width 3840 and a final norm of 3840; each of its 48 layers has four norms of 3840, one scalar, bias-free q, k, v, o matrices (16 query heads; 8 key-value heads of width 256 in a sliding layer, and in a full layer, every sixth, 1 of width 512 with no v matrix), two norms of the head width (q and k) and a bias-free gated feed-forward of three 3840 x 15360 matrices; the two aggregate heads each have a bias. The tensor counts (4,349 for LTX, 825 for Wan 1.3B) are read from the manifests and are data, not derived here.
<!-- /worked -->

The LTX-2.5 transformer comes to 21.0 billion parameters by its manifest, 4349 tensors, where the checkpoints are named 22B, a round figure the repository does not derive. Sixty-one percent of the weights serve the picture, 27 the sound and its coupling, and 12 the connectors, the timestep networks and the ends. Wan 1.3B is 1.42 billion by its own manifest of 825 tensors, and Gemma-4's text tower, embedding table included, is 11.9 billion. What one token stands for differs by a factor of eight.

<!-- worked: video_generation.latent_budget_table -->
What one transformer token stands for in the default clip of each model. LTX-2.5: a token is one latent cell of 128 channels covering 8 frames x 32 x 32 pixels. Wan: a latent cell of 16 channels covers 4 frames x 8 x 8 pixels and a token is a 2 x 2 patch of cells, 64 values, covering 4 x 16 x 16 pixels. 'RGB / latent' is the RGB values of the clip over its latent values.

| model | pixels per token (frames x rows x columns) | values per token | tokens | latent values | RGB / latent |
|---|---|---:|---:|---:|---:|
| LTX-2.5 | 8 x 32 x 32 | 128 | 6,144 | 786,432 | 181.5 |
| Wan | 4 x 16 x 16 | 64 | 32,760 | 2,096,640 | 46.3 |
<!-- /worked -->

LTX's autoencoder compresses the 121-frame clip 181 times in values and gives the transformer 6144 tokens of 128 values; Wan's compresses 46 times and gives it 32 760 tokens of 64. At the same clip length of about five seconds, Wan's transformer attends over 5.3 times as many tokens, and work, not time, is the way to compare what each asks of a machine. The forwards of a clip are the steps times the passes of a step, and the cost of each forward is the video-stream count of the first section.

<!-- worked: video_generation.clip_work_table -->
The denoiser's forwards for a clip of about five seconds: LTX-2.5 at 768 x 512, 121 frames, 24 frames per second (6144 tokens); Wan 1.3B at 832 x 480, 81 frames, 16 frames per second (32760 tokens). A step runs one forward for the conditional prediction plus one for each guidance term that is switched on (a negative-prompt term when the scale is not 1, an STG term when its scale is not 0, an isolated-modality term when its scale is not 1). The forward is the video-stream FLOPs of the table above; the sound stream and the VAE decode are not counted. The last column assumes every forward reads every weight of the video stream once, at one byte per weight for LTX-2.5 (8-bit) and four for Wan (32-bit, the default).

| setup | steps | passes per step | forwards | FLOPs per forward (10^12) | FLOPs of the clip (10^15) | weight bytes read (10^9) |
|---|---:|---:|---:|---:|---:|---:|
| LTX-2.5 distilled, one stage, no guidance | 8 | 1 | 8 | 176.5 | 1.41 | 103 |
| same, with a negative-prompt term (scale 3) | 8 | 2 | 16 | 176.5 | 2.82 | 206 |
| same, with the reference preset (cfg 3, stg 1, modality 3; needs sound) | 8 | 4 | 32 | 176.5 | 5.65 | 413 |
| Wan 1.3B, guidance 5 | 50 | 2 | 100 | 283.0 | 28.30 | 568 |

The Wan clip needs 20.0 times the arithmetic of the distilled LTX clip of the same length.
<!-- /worked -->

The distilled LTX clip is 1.41 PFLOP, 8 forwards, and each of its weights is used in $2N = 12\,288$ FLOPs per forward, where $N = 6144$. Adding a negative-prompt term doubles it, and the reference preset's four passes quadruple it. The Wan clip is 28.3 PFLOP, 20 times as much, from two factors that compound: 12.5 times the forwards, 100 against 8, and 1.6 times the arithmetic per forward, where 5.3 times as many tokens meet a transformer about nine times smaller. Where the arithmetic goes is as informative as the total: for Wan 70 percent of each block is attention scores, which the width of the model does not touch and only the token count does, and a larger autoencoder stride of the kind LTX uses is a saving in the quantity that enters squared.

## What the code does not yet do

A chapter that praises a design owes a list of what the code leaves out, and each item below was checked against the source. The token-count schedule of LTX exists, matches the reference to a tolerance below $10^{-6}$ in its parity test, and is reached only by the random-weight configuration, because the real path takes the distilled table. No non-distilled LTX checkpoint is configured, so the shifted ramp has not generated a clip with real weights. LTX-2.3 is loadable but not runnable: its text encoder is Gemma 3, and `crates/gemma4` has no Gemma 3 path, as the doc comment of `LtxDitConfig::ltx23_22b` says. Training exists for both streams, gradient-checked, but on synthetic data and a simplified geometry: `make_flow_batch` in `crates/ltxv/src/modelgrad.rs` builds the velocity target $\boldsymbol{\varepsilon} - \mathbf{x}_0$ of Volume VIII, with positions that are a single row of frames with height and width pinned and the same $\sigma$ as the timestep of every token, so the trainer meets neither the pixel-unit positions nor the per-token timesteps of this chapter. Wan's trainer, which the same comment describes, reads captioned clips from disk. Wan has no windowing, no conditioning on a first frame, and no sound; its clip is one pass, of 81 frames by default. And the sampler described above for LTX, with its fixed table and its ancestral noise, is not what Wan runs: Wan takes a multistep flow-matching solver, UniPC by default (a predictor-corrector scheme that reuses the last few velocity predictions to take a higher-order step, so that it needs fewer steps than a first-order Euler loop), 50 steps of two forwards each.

The lesson of the whole record is the one of the opening: a clip is made of many parts that each look right alone, and what failed each time was the agreement between two of them, the position and its unit, the token and its timestep, the window and the one before it.

## Check your understanding

1. **Find the error.** "A 16-frame-latent clip scores 16 times as many attention pairs as a single picture, because it has 16 times the tokens." Correct it, and say what the ratio is for 61 latent frames.
2. **Predict.** A 768 by 512 clip is lengthened from 121 to 241 frames. By what factor does the self-attention term of a block grow, by what factor the linear terms, and which of the two does the crossover $N^\star = 3d + d_{\mathrm{ff}}$ say dominates for LTX at 11 904 tokens?
3. **Explain.** Why does LTX normalise the frame axis by 20 seconds and the others by 2048 pixels, and what would a model told only the frame index see if the clip were generated at 12 and at 24 frames per second?
4. **Find the error.** "A frozen token is pinned to its clean value after every step, so the network does not need to be told it is frozen." Say what goes wrong, using the symptom of commit `bf00210e5`.
5. **Predict.** A token held at strength 0.8 is in a clip at the step $\sigma = 0.725$. What timestep does the network get for it, and what is its clean estimate after the blend if $\mathbf{x} = 3$, the velocity is 2 and its clean content is 10?
6. **Explain.** Why can a 601-frame clip not be built from windows of the form $1 + 8k_i$ that are decoded independently, and what makes the carried-context plan sum exactly?
7. **Find the error.** "The first window of the plan for 24 frames per second is 34 latent frames, the full budget." Why is it 32?
8. **Predict.** The negative-prompt scale of LTX is set to 3 and no other guidance scale is changed. How many forwards does one step run, and why does the isolated-modality term add no pass?

## Summary

A clip stays consistent inside one pass because its tokens attend across time as freely as across space, which costs pairs that grow with the square of the clip, and because each token carries a position in physical units, seconds and pixels in LTX and integer axes in Wan, whose scale a model cannot recover if it is wrong. The sampler's schedule belongs to the model: a token-count shift in the derivation, a fixed distilled table in practice, run at a quarter of the tokens and refined at full size. Conditioning is a token with timestep 0 that the sampler pins, the network is told about, and the clean estimate leaves alone, and the same mechanism holds a first frame, the carried tail of an earlier window and a given soundtrack. Sound is a second stream in every block with cross-attention in both directions on a shared clock. Long clips are windows with eight latent frames carried across each seam, planned so that the emitted frames sum exactly and, with sound, so that seams fall on whole audio tokens. Every claim of speed in the chapter is a count of FLOPs or bytes. The limit of all of it is the pass: a window cannot grow past the tokens one forward can attend over, and quality inside a window still depends on training that the repository reproduces only on synthetic data. The next chapter leaves the generation of what a camera sees for the geometry of the camera itself, how a point in space becomes a pixel and back, which is what a model that is asked to make a clip from every side will need.

## References

- Ho, Jain, Abbeel, *Denoising Diffusion Probabilistic Models*, 2020.
- Lipman, Chen, Ben-Hamu, Nickel, Le, *Flow Matching for Generative Modeling*, 2023.
- Peebles, Xie, *Scalable Diffusion Models with Transformers*, 2023.
- Su, Lu, Pan, Murtadha, Wen, Liu, *RoFormer: Enhanced Transformer with Rotary Position Embedding*, 2021.
- Dao, Fu, Ermon, Rudra, Re, *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, 2022.
- Wan Team, *Wan: Open and Advanced Large-Scale Video Generative Models*, 2025.
- HaCohen and others, *LTX-Video: Realtime Video Latent Diffusion*, 2024.

## Exercises

1. **[Calculate]** For the clip of 49 frames at 768 by 512, compute the pairs of positions scored by one head with attention frame by frame, space then time, and all at once, and the ratio of the last to the middle. Check against the table of the first section.
2. **[Calculate]** The crossover $N^\star = 3d + d_{\mathrm{ff}}$ for Wan 14B ($d = 5120$, $d_{\mathrm{ff}} = 13\,824$) and the share of the 75 600-token clip's block that is attention scores. Is the clip above or below the crossover, and by how much?
3. **[Calculate]** The frame-axis position $\bar p$ of latent frame 5 of a clip at 30 frames per second, and the span in seconds of that latent frame at 12 frames per second.
4. **[Implement]** Reproduce the table of `video_generation.schedule_shift_table`: $\mu(N)$, the four-step ramp, the shift, and the terminal stretch to 0.1. Then compute the four-step schedule at $N = 2048$ and check that its last non-zero value is 0.1.
5. **[Calculate]** A token held at strength 0.8 (mask 0.2) at $\sigma = 0.725$ with $\mathbf{x} = 3$, velocity 2 and clean content 10: its timestep, its clean estimate, and the estimate after the blend.
6. **[Implement]** Reproduce `video_generation.frozen_loop`: implement `denoise_clip` of the pseudocode and run the three tokens through the eight steps of the distilled table with the constant velocity 1. Verify that the free token ends at 1 and the pinned one never moves.
7. **[Implement]** Reproduce `video_generation.window_table` with your own `window_plan`, and add the plan for 361 frames at 768 by 512. Verify that the emitted frames sum to the request and that each window's token count is at most 13 200.
8. **[Calculate]** A clip of 97 frames at 30 frames per second: its audio tokens, the mel frames and seconds of its decoded sound, and how much shorter than the picture that is. What is the quantum $q$ at 30 frames per second?
9. **[Implement]** Reproduce `video_generation.seam_toy` from the stream name and seed printed with it: 2000 trials, nine normals each, and the mean velocity error for $c = 1, 2, 3, 4, 8$. Compare with the predicted column.
10. **[Calculate]** Wan's tokens for 49 frames at 640 by 384 pixels, and the attention pairs of one head.
11. **[Investigate]** Using the FLOP model of the first section, find the clip length in latent frames at 768 by 512 at which LTX's self-attention scores equal its token-side linear layers, and say whether a single window of the plan can reach it.
12. **[Investigate]** Count the forwards of one LTX step for each combination of the three guidance terms (negative prompt, STG, isolated modality), and the total FLOPs of an 8-step clip for each, using the forward cost of the table of the clip's work.

## Key terms

- **spatiotemporal attention**: self-attention in which the tokens of a clip, placed on a grid of frame, row and column, attend to one another across all three axes. <!-- concept: spatiotemporal-attention -->
- **video diffusion**: diffusion or flow-matching generation on the latent grid of a video: a transformer predicts the velocity of every token of a noised clip, with a position on each of three axes and a timestep for each token, and a sampler turns the noise into a clip's latent. <!-- concept: video-diffusion -->
- **joint audio-video generation**: one transformer that denoises a video latent and an audio latent together, each in a stream of its own, coupled in every block by cross-attention in both directions on a shared clock of seconds. <!-- concept: joint-audio-video-generation -->

## Answers

**Recall questions.**

1. 121 frames give $1 + 120/8 = 16$ latent frames and 25 frames give $1 + 24/8 = 4$. A request for 24 frames is refused, since $23$ is not a multiple of 8.
2. $N^2$ scores per head. The scores take $2N^2 d$ FLOPs and the weighted sum of values the same, $4N^2 d$ together.
3. 64 pairs. A pair rotates by the position on the axis that owns it times its own rate.
4. $\sigma' = 3 \cdot 0.5/(1 + 2 \cdot 0.5) = 0.75$, and $\mu = \ln 3 = 1.0986$, since $k = e^\mu$.
5. The slope of the line that fits all of them, the velocity of the points, which one point cannot give, and which averages the noise of the $c$ points down as $c$ grows.

**Check your understanding.**

1. The pairs of one picture are $384^2$ and of the clip $(16 \cdot 384)^2$, so the ratio is $16^2 = 256$, the square of the ratio of the tokens. For 61 latent frames it is $61^2 = 3721$.
2. 121 frames are 16 latent frames and 241 are 31, so the tokens go from 6144 to 11 904, a factor of 1.94. The linear terms grow by 1.94 and the self-attention term by its square, 3.75. At 11 904 tokens the crossover $3 \cdot 4096 + 16\,384 = 28\,672$ is above, so the linear layers still dominate.
3. Because the position is a physical quantity (seconds and pixels), normalised by the largest value each axis is calibrated to. A frame index alone would put frame 8 at the same position at both rates, while in seconds the second clip's frame is at half the position of the first.
4. The blend pins the output only. Told nothing, the network sees a clean token labelled as noisy, modulates it as noise through the per-token adaLN rows, and mixes it into every free token by attention. The symptom is the one the commit names: the pinned frame decoded perfectly and every other frame was a mesh texture.
5. $t = 0.2 \cdot 0.725 = 0.145$. The estimate is $3 - 0.145 \cdot 2 = 2.71$, and the blend is $0.2 \cdot 2.71 + 0.8 \cdot 10 = 8.542$.
6. $n$ windows of $1 + 8k_i$ frames sum to $n + 8\sum k_i$, and a legal clip is $1 + 8k$, so only $n = 1$ works. With carried context only the first window keeps its leading frame and each other window emits $8u$ frames, so the sums are $1 + 8(u_0 - 1) + 8\sum_{j>0} u_j = 1 + 8k$.
7. At 24 frames per second the quantum is 3: a window followed by another must advance, by its latent frames minus its 8 carried, a multiple of 3. A first window of 34 would advance 26. The largest that keeps the advance a multiple of 3 is 32, with advance 24.
8. A negative-prompt scale of 3 is not 1, so a step runs the negative pass besides the conditional one, 2 forwards. The isolated-modality pass runs only when its scale is not 1, and the default is 1, so it adds nothing.

**Exercises.**

1. Seven latent frames of 384 tokens: frame by frame $7 \cdot 384^2 = 1\,032\,192$; space then time $1\,032\,192 + 384 \cdot 49 = 1\,051\,008$; all at once $2688^2 = 7\,225\,344$; ratio of the last to the middle 6.9.
2. $3 \cdot 5120 + 13\,824 = 29\,184$. The clip of 75 600 tokens is 2.6 times above it, and attention scores are 72 percent of the block.
3. Latent frame 5 covers pixel frames 33 to 41, at 30 frames per second 1.1 to 1.3667 seconds, the midpoint is 1.2333 and $\bar p = 2 \cdot 1.2333/20 - 1 = -0.8767$. At 12 frames per second the span is 2.75 to 3.4167 seconds.
4. See the table: $\mu(2048) = 1.3167$, $e^\mu = 3.731$, and the schedule is $1, 0.8344, 0.5732, 0.1, 0$.
5. Timestep $0.2 \cdot 0.725 = 0.145$; estimate $3 - 0.145 \cdot 2 = 2.71$; after the blend $8.542$.
6. The trace is the table of the block: free token $1 + \sigma_i$ before step $i$, ending at 1; pinned token 5 throughout; half-held token ending at 7.795785.
7. 361 frames give $(361 - 1)/8 = 45$ groups, plan $[(0, 34, 0), (8, 12, 265)]$: window 0 emits 265 frames, window 1 emits $8 \cdot 12 = 96$, together 361; the tokens are $34 \cdot 384 = 13\,056$ and $20 \cdot 384 = 7680$.
8. $N_{\mathrm a} = \mathrm{round}(97 \cdot 25/30) = \mathrm{round}(80.83) = 81$; mel frames $4 \cdot 81 - 3 = 321$, 3.21 seconds; the picture lasts 3.2333, so the sound is 0.0233 seconds shorter. The quantum at 30 frames per second is 3.
9. The table of the block; the means for $c = 1, 2, 3, 4, 8$ are 2.006, 0.222, 0.114, 0.071, 0.025 against predictions 2.025, 0.226, 0.113, 0.071, 0.025.
10. $13 \cdot 24 \cdot 40 = 12\,480$ tokens, and $12\,480^2 = 155\,750\,400$ pairs.
11. Setting $4N^2 d = 28\,d^2 N$ gives $N = 7d = 28\,672$ tokens, 74.7 latent frames of 384; a window holds at most 34, so no single window reaches the crossover at this size.
12. One plus the number of switched-on terms: none 1, negative prompt 2, STG 2, modality 2, any two 3, all three 4; each forward is 176.5 x 10^12 FLOPs so an 8-step clip is 1.41, 2.82, 4.24 and 5.65 PFLOP for 1, 2, 3 and 4 passes.
