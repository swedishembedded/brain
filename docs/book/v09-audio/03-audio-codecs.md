<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Neural Audio Codecs

*How can audio become a stream of discrete tokens a language model can predict?*

One second of speech at 24 kHz is 24,000 numbers of 16 bits each, 384,000 bits. The codec that brain runs for speech writes the same second as 12.5 columns of 16 integers, each below 2048, and a decoder turns those integers back into a waveform. How close that waveform is to the original is what the codec's training is for; what the arrangement costs takes one line per stream.

<!-- worked: audio_codecs.token_rates -->
Bits per second = frames (or samples) per second x numbers per frame x bits per number; a codebook of 2048 entries costs log2 2048 = 11 bits per code. The last column divides the bits of 24 kHz, 16-bit samples by the row's bits.

| stream | frames or samples per second | numbers per frame | bits per number | bits per second | numbers per second | 24 kHz PCM bits / row bits |
|---|---:|---:|---:|---:|---:|---:|
| raw samples, 24 kHz, 16 bit | 24000 | 1 | 16 | 384,000 | 24,000.0 | 1.0 |
| raw samples, 16 kHz, 16 bit | 16000 | 1 | 16 | 256,000 | 16,000.0 | 1.5 |
| brain's codec: 16 codebooks of 2048 | 12.5 | 16 | 11 | 2,200 | 200.0 | 174.5 |
| the same frames with 8 codebooks (illustration) | 12.5 | 8 | 11 | 1,100 | 100.0 | 349.1 |
| the same frames with 1 codebook (illustration) | 12.5 | 1 | 11 | 137.5 | 12.5 | 2792.7 |
<!-- /worked -->

The codec's stream is 174.5 times smaller in bits, and the number that matters to a language model is not the bits but the count of symbols: 200 integers a second, against 24,000 for the samples. A model that reads a sequence pays for every pair of positions it attends over, $n(n+1)/2$ of them for $n$ positions. For ten seconds of audio that is 28,800,120,000 pairs for the samples and 2,001,000 for the codec's 2000 integers, a ratio of about 14,400, and the samples are also the wrong kind of symbol: a 16-bit sample is one of 65,536 values whose neighbours differ by almost nothing, so a model asked to name the next one is asked a question whose answer is nearly the previous sample. The codec's integers are different. Each is a choice among 2048 entries of a table, made by a network trained so that the few choices it can afford carry what the decoder needs. A stream of such integers is a sentence in a language with 2048 words, which the language models of Volumes V and VI are built to continue, and this chapter is about how a waveform is turned into one and back.

Two pieces of the answer come from earlier volumes, and two are new. The encoder that shrinks 1920 samples into one column is a stack of strided convolutions, and the decoder that grows it back is a stack of transposed ones, both from Volume VII. The table of centres that turns a column into integers is the vector quantizer of Volume VIII, but one table is not enough here, and what replaces it, a chain of tables each correcting the error the previous one left, is the new idea of the chapter. Four things to bring back before they are used, with their answers at the end:

1. (Volume IX, [Chapter 1](01-sound.md), waveform and sampling rate.) At 24 kHz, how many samples are there in 80 milliseconds?
2. (Volume VIII, [Chapter 3](../v08-image-generation/03-vector-quantization.md), vector quantization.) What does a vector quantizer store for a vector $\mathbf{z}$, and which code wins when two are equally near?
3. (Volume VII, [Chapter 2](../v07-vision/02-convolution.md), receptive field and stride.) A layer has kernel $k$ and stride $s$ and sits after layers whose receptive field is $r$ and whose jump is $j$. What are the receptive field and the jump after it? How long is the output of a stride-$s$ convolution with "same" padding on an input of length $H$?
4. (Volume VII, [Chapter 2](../v07-vision/02-convolution.md), transposed convolution.) A transposed convolution with kernel $k$, stride $s$ and padding $p$ takes a length $n$ to what length, and what do overlapping stamps do?

## Eighty milliseconds in one column

The encoder's job is arithmetic on lengths before it is anything else. It must turn $L$ samples into $T$ columns with $T$ near $L$ divided by a fixed number of samples per column, the **hop** $H$ of this chapter (the hop of the spectrogram in Chapter 2 is the same idea: the number of input samples between two neighbouring outputs). A strided convolution with stride $s$ divides the length by $s$, so a stack of them divides it by the product of their strides, and the hop is that product. brain's codec uses strides 4, 5, 6 and 8 in its convolution stack, then a convolution of stride 2 after its transformer, so

$$
\begin{aligned}
H &= 4 \cdot 5 \cdot 6 \cdot 8 \cdot 2 = 1920 \text{ samples} = 80 \text{ milliseconds},\\
f_{\mathrm{fr}} &= \frac{24000}{1920} = 12.5 \text{ columns per second}.
\end{aligned}
$$

![The encoder and decoder of brain's codec as two rows of five boxes: the waveform at 24000 Hz passes strided convolutions with strides 4, 5, 6 and 8 to 25 Hz by 512, a transformer, a stride-2 convolution to 12.5 Hz by 512 and a residual quantizer giving 16 codes per column; the decoder gathers and sums the 16 codes to 12.5 Hz by 512, applies a convolution and transformer at 1024 channels, two transposed convolutions of stride 2 to 50 Hz, and four of strides 8, 5, 4 and 3 back to 24000 Hz.](img/codec-pipeline.svg)

Each layer of the stack is causal, which means it may read the past but not the future, so that a stream can be encoded as it arrives. A causal convolution of kernel $k$ and stride $s$ is a convolution with all of its zero padding on the left: $k - s$ zeros for a strided layer (so that the first window ends on the first sample and every later window ends $s$ samples further on), and $k - 1$ for a stride-1 layer. Windows are placed until the input runs out, and a window that would read past the end reads zeros there, which gives an output of $\lceil L/s \rceil$ positions. That is the "same" padding of Volume VII with the whole border on one side. The receptive field and the jump follow the recursion that chapter gave, $r_\ell = r_{\ell-1} + (k-1)\,j_{\ell-1}$ and $j_\ell = j_{\ell-1}\,s$. Here is the whole encoder, layer by layer, for one second and half a second of audio.

<!-- worked: audio_codecs.encoder_lengths -->
Each convolution is causal: zeros are laid on the left only, k - stride of them for a strided convolution and (k - 1) for a stride-1 one, and the output has ceil(length / stride) positions. 'hop' is the number of input samples between two neighbouring outputs. The receptive field is the number of samples of the clip that can change one output, rf += (k - 1) * hop before the layer, a residual branch counted once.

| layer | channels | kernel k | stride | zeros on the left | length for 24,000 samples | length for 12,000 samples | hop (samples) | receptive field (samples) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| head | 1 -> 64 | 7 | 1 | 6 | 24,000 | 12,000 | 1 | 7 |
| stage 1 residual a | 64 -> 32 | 3 | 1 | 2 | 24,000 | 12,000 | 1 | 9 |
| stage 1 residual b | 32 -> 64 | 1 | 1 | 0 | 24,000 | 12,000 | 1 | 9 |
| stage 1 down by 4 | 64 -> 128 | 8 | 4 | 4 | 6,000 | 3,000 | 4 | 16 |
| stage 2 residual a | 128 -> 64 | 3 | 1 | 2 | 6,000 | 3,000 | 4 | 24 |
| stage 2 residual b | 64 -> 128 | 1 | 1 | 0 | 6,000 | 3,000 | 4 | 24 |
| stage 2 down by 5 | 128 -> 256 | 10 | 5 | 5 | 1,200 | 600 | 20 | 60 |
| stage 3 residual a | 256 -> 128 | 3 | 1 | 2 | 1,200 | 600 | 20 | 100 |
| stage 3 residual b | 128 -> 256 | 1 | 1 | 0 | 1,200 | 600 | 20 | 100 |
| stage 3 down by 6 | 256 -> 512 | 12 | 6 | 6 | 200 | 100 | 120 | 320 |
| stage 4 residual a | 512 -> 256 | 3 | 1 | 2 | 200 | 100 | 120 | 560 |
| stage 4 residual b | 256 -> 512 | 1 | 1 | 0 | 200 | 100 | 120 | 560 |
| stage 4 down by 8 | 512 -> 1024 | 16 | 8 | 8 | 25 | 13 | 960 | 2,360 |
| tail | 1024 -> 512 | 3 | 1 | 2 | 25 | 13 | 960 | 4,280 |
| after the transformer: down by 2 | 512 -> 512 | 4 | 2 | 2 | 13 | 7 | 1920 | (attention) |
<!-- /worked -->

The first column of shapes is what a diagram would show. The channel count doubles at each of the four stages, 64 to 1024, while the length falls by the stride, so a stage does not shrink the number of values by its stride but by half of it, and the last convolution brings 1024 channels down to the 512 that the transformer reads. A residual branch at the start of every stage (a convolution of kernel 3 to half the width, one of kernel 1 back) adds depth at the stage's own rate without changing it. At the end of the convolutions the clip of 24,000 samples is 25 positions, 25 Hz, and what the stack could see to compute one of them is 4,280 samples, about 178 milliseconds. That is the most any column's value can depend on the signal, from the convolutions alone; the transformer that follows can look further back.

Two consequences of the ceiling follow. The first is that five nested ceilings are one: $\lceil \lceil n/4 \rceil / 5 \rceil = \lceil n/20 \rceil$, and so on down the stack. To see it, take an integer $t$ and ask when $\lceil \lceil n/a \rceil / b \rceil \le t$. The inner ceiling is at most $bt$ exactly when $n/a \le bt$, that is when $n \le abt$, which is exactly when $\lceil n/(ab) \rceil \le t$; the two sides are below the same integers and so are equal. The number of columns of a clip is therefore $T = \lceil n/1920 \rceil$ whatever the order of the strides, and the table has $T = 13$ for the second where the arithmetic of rates said 12.5. The second consequence is for the decoder, which does not know $n$:

<!-- worked: audio_codecs.frame_alignment -->
A clip of n samples is cut by the five strides 4, 5, 6, 8 and 2 in turn, each output having ceil(length / stride) positions; the decoder returns 1920 samples for every frame.

| n (samples) | seconds | frames T | decoded 1920 T | extra samples | extra (ms) |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.0000 | 1 | 1,920 | 1,919 | 80.0 |
| 1,920 | 0.0800 | 1 | 1,920 | 0 | 0.0 |
| 1,921 | 0.0800 | 2 | 3,840 | 1,919 | 80.0 |
| 12,000 | 0.5000 | 7 | 13,440 | 1,440 | 60.0 |
| 24,000 | 1.0000 | 13 | 24,960 | 960 | 40.0 |
| 24,001 | 1.0000 | 13 | 24,960 | 959 | 40.0 |
| 48,000 | 2.0000 | 25 | 48,000 | 0 | 0.0 |
| 240,000 | 10.0000 | 125 | 240,000 | 0 | 0.0 |
<!-- /worked -->

A decoder that returns 1920 samples for every column returns 24,960 samples for the clip of one second, 40 milliseconds more than it was given, and 13,440 for the half second, 60 more. Whoever calls it has to remember $n$ and cut the output, or live with a clip that has grown. Only clips whose length is a multiple of 1920 round-trip to their own length, and the ones with the largest surplus are those one sample past a multiple.

## One table is not enough

The encoder's last column is 512 numbers, and the quantizer must turn it into integers. A table of $K = 2048$ centres turns a vector of any width into 11 bits, and that is the budget: 11 bits per column per table. To spend more bits per column with one table, the table must grow exponentially, $2^b$ entries for $b$ bits, and it must be searched in full for every column of every clip. Sixteen codes of 11 bits is 176 bits per column, which a flat table would need $2^{176}$ entries to express. There is no such table.

The way out is old, from the speech coders of the early nineteen-eighties, and it needs no new tool. Quantize the column with a small table. The error that is left, the **residual**, is a vector too, smaller than the one you started with, and it too can be quantized, by a second table trained on the residuals the first one leaves. Repeat. The reconstruction is the sum of the centres chosen, and each stage spends its own bits on what the stages before it missed. This is **residual vector quantization**: a chain of $Q$ codebooks $\mathbf{C}^{(1)}, \dots, \mathbf{C}^{(Q)}$ of $K$ entries each, where stage $q$ quantizes the residual $\mathbf{r}^{(q-1)}$ that stages $1, \dots, q-1$ left, with $\mathbf{r}^{(0)} = \mathbf{z}$:

$$
\begin{aligned}
k_q &= \arg\min_{m}\ \lVert \mathbf{r}^{(q-1)} - \mathbf{c}^{(q)}_m \rVert^2, \\
\mathbf{r}^{(q)} &= \mathbf{r}^{(q-1)} - \mathbf{c}^{(q)}_{k_q}, \\
\hat{\mathbf{z}} &= \sum_{q=1}^{Q} \mathbf{c}^{(q)}_{k_q},
\end{aligned}
$$

and the column is stored as the $Q$ indices $(k_1, \dots, k_Q)$, $Q \log_2 K$ bits. The first stage is the quantizer of Volume VIII, with its tie rule (the lower index wins). Here it is by hand, in two dimensions. Stage $q$ has four codes, the corners $(\pm 0.5, \pm 0.5)$ scaled by $0.5^{q-1}$, so each stage's table is the previous one shrunk by half. (The table has no entry at the origin, so a residual smaller than the grid can be pushed further off by a stage; Exercise 4 finds one. The trained tables below put their centres where the residuals are.)

<!-- worked: audio_codecs.rvq_trace -->
Stage q has the four codes (+-0.5, +-0.5) times 0.5^q, listed in the order (-,-), (+,-), (-,+), (+,+); each stage takes the nearest code of the residual left by the stages before it (a tie goes to the lower index).

Vector x = (0.62, -0.35), indices (1, 3, 0).

| stage q | index | code | sum of codes so far | residual | squared error |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | (0.500, -0.500) | (0.500, -0.500) | (0.120, 0.150) | 0.03690 |
| 2 | 3 | (0.250, 0.250) | (0.750, -0.250) | (-0.130, -0.100) | 0.02690 |
| 3 | 0 | (-0.125, -0.125) | (0.625, -0.375) | (-0.005, 0.025) | 0.00065 |

Vector x = (-0.18, 0.41), indices (2, 1, 3).

| stage q | index | code | sum of codes so far | residual | squared error |
|---:|---:|---:|---:|---:|---:|
| 1 | 2 | (-0.500, 0.500) | (-0.500, 0.500) | (0.320, -0.090) | 0.11050 |
| 2 | 1 | (0.250, -0.250) | (-0.250, 0.250) | (0.070, 0.160) | 0.03050 |
| 3 | 3 | (0.125, 0.125) | (-0.125, 0.375) | (-0.055, 0.035) | 0.00425 |
<!-- /worked -->

For the first vector the stages choose indices $(1, 3, 0)$ and the sum of their centres is $(0.625, -0.375)$, which is $0.005$ and $0.025$ away from $(0.62, -0.35)$. The squared error falls at every stage, 0.0369, 0.0269 and 0.00065. Six bits (three times two) were spent, from three tables of four entries that the search compared the vector with in $3 \times 4 = 12$ distances; a flat table with the same 64 cells would need 64. The reconstruction is a point of a grid, but the grid is not stored anywhere: it is the set of all sums of one entry from each table, which is why the number of cells is $K^Q$ while the number of entries that must be stored and searched is only $QK$.

```text
function residual_encode(z: array<real>[d], books: list<array<real>[K, d]>) -> list<int>
    require len(books) >= 1, "a residual quantizer needs at least one stage"
    r = copy(z)
    codes = []
    for C in books
        dist = [sum((r[j] - C[m, j])^2 for j in 0 .. d) for m in 0 .. K]
        k = argmin(dist)                # the lower index wins a tie
        append k to codes
        r = r - C[k, :]                 # the residual the next stage sees
    return codes

function residual_decode(codes: list<int>, books: list<array<real>[K, d]>) -> array<real>[d]
    require len(codes) == len(books), "one index per stage"
    zhat = zeros(real, [d])
    for q in 0 .. len(books)
        zhat = zhat + books[q][codes[q], :]
    return zhat
```

The decoder is a gather and a sum, nothing more, and that is the cheap half. The numbers of the table below show how the search cost grows. The toy is the quantizer of the next section; the second pair of rows is brain's codec, whose search runs on the host in plain loops (`rvq_encode` in `crates/mimi/src/model.rs`).

<!-- worked: audio_codecs.search_vs_cells -->
A residual quantizer compares a vector with K entries at each of its stages; a flat codebook with the same cells would compare it with all of them. Distances are counted as d multiply-adds per entry; the table of one stage is K d 32-bit floats.

| quantizer | width d | stages | entries per stage K | entries searched | cells K^stages | multiply-adds of one search | bytes of the tables |
|---|---:|---:|---:|---:|---:|---:|---:|
| the toy, 2 stages | 6 | 2 | 8 | 16 | 64 | 96 | 384 |
| the toy, 4 stages | 6 | 4 | 8 | 32 | 4,096 | 192 | 768 |
| brain's codec, 1 stage | 256 | 1 | 2048 | 2,048 | 2,048 | 524,288 | 2,097,152 |
| brain's codec, 16 stages | 256 | 16 | 2048 | 32,768 | 2^176 (about 9.58e+52) | 8,388,608 | 33,554,432 |
<!-- /worked -->

Sixteen stages of 2048 entries of width 256 are compared with 32,768 entries, 8,388,608 multiply-adds and 33.6 million bytes of tables for a column, and address $2^{176}$ different reconstructions. A flat table cannot be built at this size at all, and the residual chain makes the cost linear in the number of stages while the number of cells grows as a power.

## Training the stages

Trained tables are needed, and the way to train them is the one the previous volume's quantizer already used for its codebook term: move each centre to the mean of the vectors assigned to it, which is Lloyd's algorithm. Stage by stage, then. Train stage 1 on the data; compute the residuals it leaves; train stage 2 on those; and so on. The toy of this chapter has six-dimensional vectors of independent Gaussian coordinates whose scales are $1/(1+j)$ for coordinate $j$, so that most of the energy sits in the first coordinates.

<!-- worked: audio_codecs.rvq_data_check -->
Each of the six coordinates of a vector is a draw of normal() times the scale of its position; a vector takes its six draws in coordinate order and the vectors follow one another.

| coordinate j | scale 1 / (1 + j) |
|---:|---:|
| 0 | 1.0000 |
| 1 | 0.5000 |
| 2 | 0.3333 |
| 3 | 0.2500 |
| 4 | 0.2000 |
| 5 | 0.1667 |

Mean squared norm of a vector: 1.4327 on the 1024 training vectors, 1.4897 on the 512 held-out ones (the expected value is 1.4914).
<!-- /worked -->

The stages are trained exactly as the next table says, and every starting choice is named so a reader can reproduce the numbers.

<!-- worked: audio_codecs.rvq_stages -->
Training vectors from `audio_codecs.rvq_train` (seed 0xb306e50550bbf7e9), held-out vectors from `audio_codecs.rvq_test` (seed 0x2a091622dda5e634); d = 6, 4 stages of K = 8 codes (3 bits each). Stage q is trained by 20 rounds of Lloyd's algorithm on the residuals the first q - 1 stages leave, its starting centres the first 8 rows of the residual set in the order of one shuffle of 0 .. 1023 from `audio_codecs.rvq_init` (seed 0xc84eb39f91e97505) (one shuffle per stage, in stage order, the stream continuing); a round assigns every row to its nearest centre (lower index on a tie, squared distance summed over the coordinates in order), then replaces each centre that has rows by their mean, summed in row order, and leaves a centre with no rows where it is. The codes are the assignments to the final centres. Errors are mean squared norms of the residual; the gain is 10 log10 of the mean squared norm of the data over the error, on the training set; codes used and the perplexity are those of the stage's training assignments. Numbers are printed with the digits that agree across the run and four nudged copies (first coordinate of the first starting centre of every stage and of the flat run changed by +-1e-12 and +-3e-12).

| stages sent | bits per vector | training error | error / previous | held-out error | gain (dB) | codes used | perplexity of the codes |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 (nothing sent) | 0 | 1.4327 |  | 1.4897 |  |  |  |
| 1 | 3 | 0.4275 | 0.2984 | 0.4547 | 5.25 | 8 / 8 | 7.43 |
| 2 | 6 | 0.2248 | 0.5257 | 0.2486 | 8.04 | 8 / 8 | 7.91 |
| 3 | 9 | 0.1332 | 0.5925 | 0.1548 | 10.32 | 8 / 8 | 7.96 |
| 4 | 12 | 0.0793 | 0.5956 | 0.0944 | 12.57 | 8 / 8 | 7.91 |
<!-- /worked -->

![Mean squared error of the trained residual quantizer after 0 to 4 stages (3 bits each), on training and held-out vectors, with the error of one flat codebook of 64 centres at 6 bits. The curves fall steeply at the first stage and then more slowly, and the flat codebook lies below the two-stage point on training vectors and close to it on held-out vectors.](img/rvq-stage-error.svg)

The first stage removes the most: the error falls to 0.2984 of its starting value, 5.25 decibels, and every later stage removes a smaller share of what is left: the error that remains after a stage, as a fraction of the error before it, is 0.5257, 0.5925 and 0.5956 for stages two to four, so those stages remove 47, 41 and 40 per cent. Plausibly the first stage finds the one coordinate of large scale and the later ones face a flatter residual, where a table of 8 centres in six dimensions has less to find; the table does not say so, only the ratios do. Every stage used all eight of its codes, with perplexities near 8 (7.43 to 7.96), so no code died in this small case. The held-out error follows the training error closely: the gap is 0.0272 at the first stage and 0.0151 at the fourth, which is 6 percent of the training error at the first stage and 19 percent at the fourth. The gap in proportion grows as the stages add centres to fit what is left, but even at 12 bits the tables have learned what the distribution looks like and not only the particular vectors.

Is the chain a good use of its bits, against one table with the same number of cells? The comparison is the next table, with 64 cells in both.

<!-- worked: audio_codecs.rvq_flat -->
The flat codebook is 64 centres trained by the same 20 rounds on the same training vectors, its starting centres the first 64 rows of one shuffle from `audio_codecs.rvq_flat` (seed 0x518abc8fb18d2c27). 'entries searched' is the number of centres one vector is compared with; 'cells' is the number of different reconstructions the quantizer can give.

| quantizer | entries searched | cells | training error | held-out error |
|---|---:|---:|---:|---:|
| residual, 2 stages of 8 | 2 x 8 = 16 | 64 | 0.2248 | 0.2486 |
| one flat codebook | 64 | 64 | 0.1844 | 0.2357 |
| residual, 4 stages of 8 | 4 x 8 = 32 | 4096 | 0.0793 | 0.0944 |
<!-- /worked -->

The flat table is better, 0.1844 against 0.2248 on the training vectors and 0.2357 against 0.2486 on held-out vectors, and it searches four times as many entries. The chain loses a little at equal cells and wins at equal search. A flat codebook with the four-stage chain's 4096 cells would need 4096 centres, more than there are training vectors, where the chain reaches 0.0793 with 32 entries searched. That is the trade the codec makes: a small loss of fit per bit in exchange for a search that is linear in the bit rate.

brain does not train these tables. Its importer (`crates/mimi/src/import.rs`) reads each codebook of the released checkpoint as two stored tensors per layer, `embed_sum` with one row per entry and `cluster_usage` with one number per entry, and divides them, `table = embed_sum / max(cluster_usage, 1e-5)`. Read as the names say, that is the mean step of Lloyd's algorithm kept as moving statistics through training: a sum of the vectors assigned to an entry divided by how many were assigned is their mean. The importer's comment says the division matches the reference's `EuclideanCodebook.decode`, and a unit test in the file pins it on tables small enough to compute by hand (sums with rows $(1, 2)$, $(3, 4)$, $(5, 6)$ over counts $1, 2, 5$ give rows $(1, 2)$, $(1.5, 2)$, $(1, 1.2)$, the test's expected values).

## Splitting off a first stage

The codec's sixteen codebooks are not one chain. The crate's header calls the structure a split residual vector quantizer, and `rvq_encode` shows what that means: the encoder's 512-wide column is projected to 256 numbers by one matrix for the first group, which has a single codebook, and by a second matrix for the second group, which is a residual chain of the other fifteen. The two projections both read the same column, and the two groups quantize independently, so the semantic group's code is not computed from the residual of anything and the acoustic chain does not start from what the semantic code left. The decoder reverses it: it gathers the 256-wide rows, sums the fifteen acoustic ones, passes the sum and the single semantic row through their own $256 \to 512$ projections and adds the results. Because a projection is linear, projecting the sum is the same as summing the projections, so a chain of fifteen stages needs one matrix product and not fifteen (`Codec::decode`).

The reason to split is what the first code should mean. In the design the crate follows, the first codebook is trained with an extra loss that pushes it toward something a speech-understanding model would also compute, so that its integer says roughly which sound is being made, and the other fifteen add the detail of how it is said. That is a claim about how the released tables were trained and brain does not train them: the crate's README says the distillation is not implemented. What the code does confirm is the shape. The tokens of one column are not interchangeable, the first is a different kind of thing from the rest, and a model that generates them has reason to treat it differently, which is where Chapter 6 picks the story up. The crate that generates speech from text, `crates/qwen3tts`, says in its header that its large decoder predicts the first codebook of each column and a small second network fills in the other fifteen from it.

Two details of the code show that the crate and the checkpoint agree on the sizes and that its configuration is untidier than its arithmetic. The decode path reads every table with 2048 rows of 256 numbers, and the checkpoint test in `crates/mimi/tests/it/decode.rs` asserts that the first group's table and the last of the fifteen are both $2048 \times 256$ (a test that skips without the real checkpoint, which was not available when this chapter was written, so the assertion could not be run for it). The configuration parser, though, also reads a field `semantic_codebook_size` from the decoder's configuration with the default 4096, and nothing in the crate reads it; the encoder searches `codebook_size` rows in both groups. The field says something different from the arithmetic the crate does and that the test asserts, and it is a leftover.

## Growing a column back into a clip

The decoder must undo a factor of 1920, and it has to do so with the same operation the encoder used, run backward in length: a transposed convolution, which takes $n$ columns to $(n-1)s + k$ (minus the padding) by stamping its kernel at every $s$-th position and adding where stamps overlap. brain's decoder does it in two groups of stages. First two transposed convolutions of kernel and stride 2 (so the stamps do not overlap), each followed by a ConvNeXt block, a depthwise convolution of kernel 7, a normalisation and a pair of pointwise layers, take 12.5 columns a second to 50. Then four SEANet blocks of the encoder's family take 50 to 24,000 with strides 8, 5, 4 and 3, each a transposed convolution whose kernel is twice its stride, followed by three residual units with dilations 1, 3 and 9 (kernel 7, so that a few units see a long stretch at the block's rate). The decoder's strides are not the encoder's run backward: $2 \cdot 2 \cdot 8 \cdot 5 \cdot 4 \cdot 3$ and $4 \cdot 5 \cdot 6 \cdot 8 \cdot 2$ are two factorisations of the same 1920, and only the product has to agree.

A transposed convolution whose kernel is twice its stride $s$ leaves $k - s = s$ samples of tail that later columns would still have to add to. The causal version crops those off the right, which makes the output exactly $s$ times as long as the input. The nonlinearity in the residual units is **SnakeBeta**, $y = x + \dfrac{1}{e^{\beta} + \varepsilon}\sin^2(e^{\alpha} x)$ with a learned $\alpha$ and $\beta$ per channel and $\varepsilon = 10^{-9}$ (the `snake` function in `crates/mimi/src/model.rs`). It is periodic, unlike a rectifier, which is the argument of the paper in the references for using it where the signal to be produced repeats, as a voiced sound does.

How much does all this cost? The count follows the layers of the configuration (the encoder's from the layer list of the previous section, which no shape test of the crate asserts, the decoder's from its blocks), with the sizes of the tensors the code reads. The decoder shapes it assumes include the ones the checkpoint test above asserts: the pre-convolution weight is $1024 \times 512 \times 3$, the first SEANet weight $1536 \times 1024 \times 7$ and the first transposed one $1536 \times 768 \times 16$.

<!-- worked: audio_codecs.codec_layout -->
The layers behind the counts below: widths, heads, kernels, and which tensors are counted. Where two matrices or norms are named they are the tensors the code reads; nothing else carries a parameter.

| component | layout | tensors counted besides the weights named |
|---|---|---|
| encoder convolutions | head 1 -> 64 (k 7); four stages with strides 4, 5, 6, 8, each a residual branch c -> c/2 (k 3) then c/2 -> c (k 1) and a strided convolution c -> 2c (k 2 x stride); tail 1024 -> 512 (k 3) | weight and bias on every convolution; the activations carry nothing |
| encoder transformer | 8 layers of width 512; attention with 8 heads of width 64 (query, key, value and output matrices, 512 x 512); feed-forward 512 -> 2048 -> 512 with two matrices | no matrix has a bias; per layer two LayerNorms (weight and bias of width 512 each, 2,048 numbers) and two layer scales (one number per channel, 1,024 numbers), so 3,072 small numbers |
| frame-rate match | one convolution 512 -> 512, k 4, stride 2 | no bias |
| quantizer projections | two of 512 -> 256, one for the semantic codebook and one for the acoustic ones | no bias |
| decoder pre-transformer | input projection 1024 -> 512, 8 layers of width 512, output projection 512 -> 1024; attention with 16 heads of width 64 (matrices 512 x 1024); feed-forward gated: three matrices of 512 x 1024 (gate, up, down) | both projections have a bias; no matrix in a layer has one; per layer two RMSNorms (weight only, 2 x 512) and two layer scales (2 x 512); one final RMSNorm (512) |
| decoder upsampling stages | two stages: a transposed convolution 1024 -> 1024 with k = stride = 2, then a ConvNeXt block of width 1024: depthwise convolution k 7, LayerNorm, pointwise 1024 -> 4096 -> 1024 | transposed convolution and all three convolutions with bias; LayerNorm weight and bias; one layer scale (1024) |
| SEANet decoder | first convolution 1024 -> 1536 (k 7); four blocks of rates 8, 5, 4, 3, each a SnakeBeta, a transposed convolution c_in -> c_out with k = 2 x rate, and three residual units (dilations 1, 3, 9) of SnakeBeta, convolution c_out -> c_out (k 7), SnakeBeta, convolution c_out -> c_out (k 1); a last SnakeBeta and a convolution 96 -> 1 (k 7) | weight and bias on every convolution; every SnakeBeta has an alpha and a beta per channel (2 x channels of its input) |
<!-- /worked -->

<!-- worked: audio_codecs.codec_counts -->
Parameters are counted from the tensors the code reads (weights and the biases it adds; the quantizer tables at 2048 entries of 256 numbers each). A convolution costs c_out c_in k / groups multiply-adds per output position and a transposed convolution c_in c_out k per input position; a matrix product costs its input width times its output width per row; the nearest-code search is counted as d multiply-adds per entry. The squared distances of the attention scores, the activations and the normalisations are not counted. Frames arrive at 12.5 per second, 25 per second inside the encoder, 50 per second after the upsampling stages. Each total is the sum of the unrounded rows, so a printed total can differ by 0.1 from the sum of the printed rows.

| component | parameters | multiply-adds per second of audio (millions) |
|---|---:|---:|
| encoder convolutions (head, 4 stages, tail) | 12,628,256 | 2,016.2 |
| encoder transformer, without attention scores | 25,190,400 | 629.1 |
| frame-rate match (512 -> 512, k 4, stride 2) | 1,048,576 | 13.1 |
| quantizer input projections (2 x 512 -> 256) | 262,144 | 3.3 |
| quantizer tables (16 x 2048 x 256) and search | 8,388,608 | 104.9 |
| **Encoder total** | 47,517,984 | 2,766.5 |

| component | parameters | multiply-adds per second of audio (millions) |
|---|---:|---:|
| dequantizer tables (16 x 2048 x 256) | 8,388,608 | 0 |
| dequantizer output projections (2 x 256 -> 512) | 262,144 | 3.3 |
| pre-convolution (512 -> 1024, k 3) | 1,573,888 | 19.7 |
| pre-transformer without attention scores (8 layers) and projections | 30,427,136 | 380.1 |
| upsampling stages (2 x transposed convolution + ConvNeXt block) | 21,006,336 | 708.3 |
| SEANet decoder: first convolution | 11,011,584 | 550.5 |
| SEANet block 1: up by 8, 1536 -> 768 channels | 33,047,808 | 6,606.0 |
| SEANet block 2: up by 5, 768 -> 384 channels | 6,496,896 | 8,257.5 |
| SEANet block 3: up by 4, 384 -> 192 channels | 1,478,976 | 8,257.5 |
| SEANet block 4: up by 3, 192 -> 96 channels | 333,984 | 6,193.2 |
| SEANet tail (96 -> 1, k 7) | 865 | 16.1 |
| **Decoder total** | 114,028,225 | 30,992.3 |
<!-- /worked -->

The decoder has 114,028,225 parameters and the encoder 47,517,984, and the decoder needs 30,992.3 million multiply-adds for every second of audio against 2,766.5 million for the encoder, about eleven times as many. The decoder's work is nearly all in the four SEANet blocks, 29,314.2 million of the 30,992.3, although they hold only 41,357,664 of its 114 million parameters: a weight in these blocks is applied at 400, 2000, 8000 or 24,000 positions a second, where a weight of the pre-transformer is applied at 12.5. A block costs about its output width squared times its output rate. That is why block 3, which has half the channels of block 2 and four times the rate, costs exactly what block 2 does (8,257.5 million in both rows): $384^2 \cdot 2000 = 192^2 \cdot 8000$. The counts leave out the attention scores, which depend on the length of the clip and not on the second:

<!-- worked: audio_codecs.attention_work -->
Per query and key a layer spends 2 x heads x head width multiply-adds (one product for the score, one for the weighted value). The encoder has 8 layers of 8 heads of width 64 (2 x 512 per query and key per layer) and its attention, as the code runs it, sees every earlier frame, t (t + 1) / 2 pairs for t frames; the decoder has 8 layers of 16 heads of width 64 (2 x 1024) and sees at most the 72 most recent frames, the sum over frames i of min(i + 1, 72) pairs. Frames are ceil(seconds x 24000 / 960) at 25 Hz and ceil(seconds x 24000 / 1920) at 12.5 Hz.

| seconds | encoder frames (25 Hz) | encoder attention multiply-adds | decoder frames (12.5 Hz) | decoder attention multiply-adds |
|---:|---:|---:|---:|---:|
| 1 | 25 | 2,662,400 | 13 | 1,490,944 |
| 5 | 125 | 64,512,000 | 63 | 33,030,144 |
| 10 | 250 | 257,024,000 | 125 | 105,578,496 |
| 30 | 750 | 2,307,072,000 | 375 | 400,490,496 |
<!-- /worked -->

The decoder's attention sees at most the 72 most recent columns, so its cost grows in proportion to the clip once the clip is longer than 5.76 seconds. The encoder's, as the code runs it, sees every earlier column and grows with the square of the clip. The next sections say why the two differ, and the history of the difference is the most instructive thing in the crate.

## A window that was parsed and never applied

Every transformer of this codec was trained with a sliding-window causal mask: a query at position $i$ may attend to keys $j \le i$ with $i - j$ below the window, 72 columns (5.76 seconds) for the decoder's transformer and 250 for the encoder's. brain's decoder, when it was first ported, did not apply it. `CodecConfig::sliding_window` was parsed from the checkpoint's JSON, correctly, and nothing in `Codec::transformer` read it; the attention was plain causal. Commit `0893002c3` (11 August 2026) found it while working on another model that shares the function, confirmed against the reference that the mask applies on every forward call and not only in chunked decoding, and added a windowed attention kernel that is identical to plain causal attention whenever the window is at least the clip length. Knowledge #43 tells the story: it had been invisible because the tests decoded at most 24 columns, comfortably inside a window of 72, where the two masks are the same function.

The lesson was only half learned the first time, and the second half is a general one. The crate has two implementations of the decoder's transformer, one on the device and one in plain loops for the streaming decoder (`crates/mimi/src/decode_stream.rs`). The commit fixed the first. Commit `582d8216d` (4 September 2026) found that the host implementation scored every key $0$ to $i$, past frame 72 and silently: no panic, no wrong length, a waveform that drifted from the reference the further it got from the first window. The only test comparing the two ran at 16 frames, under the window, where the masks agree by construction. The fix applies the window in the host loop (also making its attention cost proportional to the window instead of the square of the clip), treats a window of 0 as unbounded (a literal 0 masked every key including the diagonal and turned the softmax into a uniform average over the whole row, future frames included), and adds `crates/mimi/tests/it/long_form_window.rs`. That test decodes the same synthetic weights through both implementations at a window of 4 and requires them to agree below $10^{-4}$ at 40 columns, and requires that changing the window moves the waveform by more than $10^{-3}$ on both, so that a change that dropped the mask on both sides could not pass the agreement vacuously.

Commit `9f5a077fe`, the same day, records a second lesson about how to test it. The first version of the real-weights test decoded a clip longer than 72 columns and bounded the difference between the two implementations, and the bound failed on a correct implementation. The reason is in the decoder: the periodic activation multiplies any input difference by $e^{\alpha}$ at each of twelve residual units, so a disagreement of the two implementations grows with the length of the clip whether or not a mask is involved, and a bound on it measures the amplifier. The replacement shrinks the window instead of lengthening the clip, which puts four fifths of the queries under a truncated key set at a length where the numerical noise is already known.

The encoder has the same unapplied window now. `EncoderConfig::sliding_window` (250 columns, 10 seconds at the encoder's 25 Hz) is parsed and `enc_transformer` dispatches plain causal attention, which is why the encoder rows of the attention table grow with the square of the length. The code says it is left deliberately, since the only witness of the encode path, a golden of codes from the reference tokenizer, is far shorter than 250 columns, so a fix could not be shown to change the result before and after (`crates/mimi/src/model.rs`, the comment in `Codec::transformer`; knowledge #43). The README claims the encoder matches the reference on every code of its golden, and what the golden cannot test is a clip of more than ten seconds.

## Streaming, and the part that does not stream

The streaming decoder (`StreamingCodecDecoder` in `crates/mimi/src/decode_stream.rs`) exists so that audio can start playing before the clip is finished. Every convolution is causal, so each can carry, from one chunk to the next, exactly what a later chunk will need, and the chunked result is the one-shot result. A stride-1 convolution with dilation $\delta$ and kernel $k$ keeps the last $\delta(k-1)$ columns of its input per channel, a transposed convolution keeps the $k - s$ output columns that later columns will still add to. The state of the whole streamed part is therefore small:

<!-- worked: audio_codecs.streaming_state -->
A stride-1 causal convolution keeps its last dilation x (k - 1) input columns per input channel; a transposed convolution keeps the k - stride output columns that later frames still add to, per output channel. The two upsampling transposed convolutions have k = stride and keep nothing. 32-bit floats.

| stage of the streamed part | floats carried | bytes |
|---|---:|---:|
| upsample 1 ConvNeXt depthwise (c 1024, k 7) | 6,144 | 24,576 |
| upsample 2 ConvNeXt depthwise (c 1024, k 7) | 6,144 | 24,576 |
| first SEANet convolution (1024 channels in, k 7) | 6,144 | 24,576 |
| block 1 transposed convolution carry (c_out 768 x 8) | 6,144 | 24,576 |
| block 1 residual conv dilation 1 (c 768, k 7) | 4,608 | 18,432 |
| block 1 residual conv dilation 3 (c 768, k 7) | 13,824 | 55,296 |
| block 1 residual conv dilation 9 (c 768, k 7) | 41,472 | 165,888 |
| block 2 transposed convolution carry (c_out 384 x 5) | 1,920 | 7,680 |
| block 2 residual conv dilation 1 (c 384, k 7) | 2,304 | 9,216 |
| block 2 residual conv dilation 3 (c 384, k 7) | 6,912 | 27,648 |
| block 2 residual conv dilation 9 (c 384, k 7) | 20,736 | 82,944 |
| block 3 transposed convolution carry (c_out 192 x 4) | 768 | 3,072 |
| block 3 residual conv dilation 1 (c 192, k 7) | 1,152 | 4,608 |
| block 3 residual conv dilation 3 (c 192, k 7) | 3,456 | 13,824 |
| block 3 residual conv dilation 9 (c 192, k 7) | 10,368 | 41,472 |
| block 4 transposed convolution carry (c_out 96 x 3) | 288 | 1,152 |
| block 4 residual conv dilation 1 (c 96, k 7) | 576 | 2,304 |
| block 4 residual conv dilation 3 (c 96, k 7) | 1,728 | 6,912 |
| block 4 residual conv dilation 9 (c 96, k 7) | 5,184 | 20,736 |
| last convolution (96 channels in, k 7) | 576 | 2,304 |
| **total** | 140,448 | 561,792 |
<!-- /worked -->

140,448 floats, 561,792 bytes, for a decoder of 114 million parameters. The tests in the file compare the streamed back with itself at different chunk sizes and, since commit `2d9057de8` (4 September 2026), `crates/mimi/tests/it/decode_stream_parity.rs` compares the whole streaming decoder with the one-shot decode on a real checkpoint for chunk sizes 0, 2 and 3. That test asserts closeness and not equality, a largest sample difference below one twentieth of the largest sample of the reference and a ratio of root-mean-square levels within one fiftieth of 1, because the one-shot decode runs on whichever device is selected and the streaming one on the host's own arithmetic. That is a bound written to catch silence, a wrong scale, a wrong length or drift with chunk count, and the commit message reports the real numbers it measured at the time, which this chapter could not reproduce without the checkpoint.

What streams is the back half only. `decode_streaming_cb` computes the front, the dequantizer, the pre-convolution and the pre-transformer, once over all the codes it is given and then hands the back half one chunk at a time, so the whole code sequence must exist before the first sample comes out. The front is causal and could be streamed too, and nothing in the crate does it. The encoder does not stream at all: `Codec::encode` takes the whole clip. A reader who wants an end-to-end real-time codec has, in this crate, a streaming decode of codes that are already known, and the algorithmic delay that matters is a column, 80 milliseconds of audio before its first code exists.

## A trainer without the hard half

The crate's name for what it trains is a reconstruction trainer, and `crates/mimi/src/recon.rs` says in its first lines what it is not. A codec like this is trained in the literature with a reconstruction loss (the waveform, and spectrograms at several scales), adversarial losses from discriminators of the waveform at several periods and scales, a feature-matching loss from the discriminators' hidden layers, and, for the first codebook, the distillation above. The file implements the first of these in a toy: a causal convolution from one channel to $c$, a quantizer, and a causal convolution back. The quantizer is the frozen table below, and it is never trained: there is no codebook loss and no mean step in the crate.

<!-- worked: audio_codecs.demo_codebook -->
Entry (b, j) of the codebook the trainer starts and keeps, for 8 codes of width 4: ((31 b + 17 j) mod 200) / 100 - 1.

| code b | j = 0 | j = 1 | j = 2 | j = 3 |
|---:|---:|---:|---:|---:|
| 0 | -1.00 | -0.83 | -0.66 | -0.49 |
| 1 | -0.69 | -0.52 | -0.35 | -0.18 |
| 2 | -0.38 | -0.21 | -0.04 | 0.13 |
| 3 | -0.07 | 0.10 | 0.27 | 0.44 |
| 4 | 0.24 | 0.41 | 0.58 | 0.75 |
| 5 | 0.55 | 0.72 | 0.89 | -0.94 |
| 6 | 0.86 | -0.97 | -0.80 | -0.63 |
| 7 | -0.83 | -0.66 | -0.49 | -0.32 |
<!-- /worked -->

<!-- worked: audio_codecs.trainer_sizes -->
The trainer's encoder is one causal convolution 1 -> c channels (kernel k, stride 1), its quantizer is the frozen table of the previous block (8 codes of width c), and its decoder one causal convolution c -> 1. Weights are c k for each convolution; the table is 8 c numbers that are never trained.

| configuration | c | k | trained weights | table entries (fixed) | receptive field (samples) | samples per frame | bits per frame | bits per second at 24 kHz |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gradient check | 4 | 3 | 24 | 32 | 5 | 1 | 3 | 72,000 |
| training test | 6 | 3 | 36 | 48 | 5 | 1 | 3 | 72,000 |
<!-- /worked -->

The sizes show why this is a toy in the sense of the chapter's first section. Its convolutions have stride 1, so there is one code for every sample, three bits a sample, 72,000 bits per second at 24 kHz: it has the quantizer of a codec and none of the strides that make one worthwhile, and its latent has 4 or 6 channels where the codec's has 512. The loss is the waveform's mean absolute error plus half its mean squared error, with a commitment term on the codes. In the book's sign convention, where $\mathrm{sign}(0) = +1$, the gradient of $\lvert d\rvert + d^2/2$ is $\mathrm{sign}(d) + d$, divided by the number of samples:

<!-- worked: audio_codecs.recon_loss -->
Loss mean(|d|) + 0.5 mean(d^2) with d = prediction - target and the book's sign (+1 at 0): gradient (sign(d) + d) / n.

| i | prediction | target | d | |d| + d^2 / 2 | gradient (sign(d) + d) / 4 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.5 | 0.4 | 0.10 | 0.1050 | 0.2750 |
| 1 | -0.2 | 0.1 | -0.30 | 0.3450 | -0.3250 |
| 2 | 0.1 | 0.1 | 0.00 | 0.0000 | 0.2500 |
| 3 | 0.9 | -0.1 | 1.00 | 1.5000 | 0.5000 |

Loss: 0.4875.
<!-- /worked -->

The third sample, where the prediction is exact, still has gradient $0.25$: $\mathrm{sign}(0) = +1$ in Rust's float `signum` as in the book, a valid choice of subgradient at the corner of $\lvert d\rvert$, but a perfect reconstruction is then not a stationary point of this loss.

The trainer comes with a gradient check, and the check teaches what a gradient check does not prove. The README reports a maximum relative error of 3.8e-3 against central differences, and the test requires below 2e-2. Until this chapter was written, however, the backward pass of the decoder ran at the continuous latent $\mathbf{z}$, and the check differentiated a function, `surrogate_loss`, whose decoder was also fed $\mathbf{z}$ (its comment: "the smooth objective the straight-through estimator differentiates"). The straight-through estimator of Volume VIII does something else: the decoder runs forward on the quantized $\mathbf{z}_q$, so its output and the gradient at its output are those of the quantized forward, and only the gradient with respect to its input is copied to $\mathbf{z}$. In `ConvReconAE::grads` the decoder's output, the gradient at it and the weight gradient of the decoder were all computed from $\mathbf{z}$, so the decoder was trained on latents it would not be given. The check agreed because it checked the code against itself. This is knowledge entry #3's lesson, finite differences prove the derivative of the forward that was emitted and not that the forward is the intended one, which that entry told of a different crate (`beta` on the wrong term). The training test, which asserts only that the loss falls below 0.85 of its start in 300 steps, could not see it either. The check that does see it needs no surrogate at all: with the assignment frozen, the real loss is smooth in the decoder's weights, so their straight-through gradient must equal a central difference of the real loss. Commit `6e1a23cdc` added that test (it failed at 0.195 against 0.068), ran the decoder on $\mathbf{z}_q$, and made the surrogate feed the decoder $\mathbf{z}$ plus a latched offset $\mathbf{q}_0 - \mathbf{z}_0$, which equals the real loss at the latch, the same construction as the image tokenizer of Volume VIII.

The history is in commit `5a134fab0` (28 June 2026), whose message lists "multiscale-mel" among the trainer's losses. The code has a multiscale log-mel function (`multiscale_mel_l1`), but it computes a number to report and has no backward pass, and nothing calls it: the file says so in a `TODO(mel-backward)`, and the crate's README says the same. Nothing outside the file calls the trainer either, so it is a verified component and not a step in any pipeline. The honest summary of what brain trains in this area is that it has the pieces of the reconstruction half, checked as the chapter says, and that the discriminators exist elsewhere in the repository (`crates/minimaxmusic3` has one, in Volume VIII's chapter on adversarial training), and are not connected to this codec.

## Tokens for a language model

The codec's output is a $[T, 16]$ array of integers, and how a language model should be shown it is a choice with no single answer. The first option is to flatten it in time order, so that 200 integers a second follow one another and each of the sixteen codes of a column is a position in the sequence. That is the simplest, the sequence is long, and an integer's meaning depends on its position within the column. The second is to predict a whole column at a time with sixteen output heads. The third, the one brain's synthesiser follows, is hierarchical: a large model predicts the first code of each column, 12.5 integers a second, and a small one fills in the other fifteen of that column from the first, so the large model's sequence is a sixteenth as long. Which to use depends on how much the residual codes depend on each other and on the one before; the codec does not say.

An **audio token** is one of those integers: an index into one stage's table, for one column, which a model treats as a symbol of a vocabulary of 2048. The definition has two parts, the table and the position, and a token's meaning depends on both: index 17 of stage 1 and index 17 of stage 2 are different symbols because the tables are different. A **neural audio codec** is the whole machine: an encoder that turns a waveform into columns, a residual quantizer that turns them into audio tokens, and a decoder that turns tokens back into a waveform, all trained together so that what the decoder returns is what the encoder was given.

## Check your understanding

1. **Find the error.** "The codec decodes a one-second clip to one second of audio, because 12.5 columns of 1920 samples are 24,000 samples." Use the table of frame alignment to say what actually happens and who must correct it.
2. **Explain.** Why is the second stage's table of a residual quantizer not a scaled copy of the first one's? Use the toy's training procedure.
3. **Predict.** brain's codec is changed to codebooks of 4096 entries. By how much do the bits per second, the multiply-adds of the search and the number of cells change?
4. **Find the error.** "The reconstruction trainer's gradient check passes with a maximum relative error of 3.8e-3, so its straight-through estimator is correct." What does the check compare, and what would it not detect?
5. **Explain.** Why does the decoder's SEANet block 4 have 333,984 parameters and still cost 6,193.2 million multiply-adds for a second, where the 21 million parameters of the upsampling stages cost 708.3 million?
6. **Predict.** A 30-second clip is encoded. By what factor is the encoder's attention work larger than for a 10-second clip, and what would it be if the encoder's window of 250 columns were applied?
7. **Explain.** Why did the first fix of the sliding window leave the streaming decoder wrong, and why could the existing test not tell?

## Summary

A neural audio codec turns a waveform into a short sequence of integers and back. Its encoder is a stack of strided convolutions whose strides multiply to the hop, 1920 samples or 80 milliseconds at 24 kHz, so that a clip of $n$ samples gives $\lceil n/1920 \rceil$ columns, with the causal padding making five nested ceilings one, and the decoder returns 1920 samples a column whatever the clip's true length. A residual vector quantizer turns each column into integers by chaining small tables, each quantizing what the previous ones left, so that $Q$ tables of $K$ entries give $K^Q$ reconstructions for $QK$ comparisons and $Q\log_2 K$ bits. The toy trained stage by stage lost a little against a flat table of the same number of cells and searched a quarter of the entries. brain's codec has sixteen tables of 2048 entries, one of them a semantic group of its own, 2,200 bits and 200 tokens a second, 47.5 million parameters in the encoder and 114 million in the decoder, and a decode that costs about eleven times the encode, nearly all of it in the four SEANet blocks, which run at the highest rates. Its history shows a parsed configuration field that nothing read, twice in the decoder and still in the encoder, and its reconstruction trainer shows a gradient check that passes because it checks the code against itself.

The token stream is what a language model can predict, and the next chapter takes the other way a speech model reads audio, as a spectrogram it must align with text it has no timing for. Chapter 6 returns to the tokens and asks how a model speaks with them.

## References

- B.-H. Juang and A. H. Gray Jr., *Multiple stage vector quantization for speech coding*, 1982. The chain of quantizers, each coding the residual of the one before.
- S. P. Lloyd, *Least squares quantization in PCM*, 1982 (written 1957). The alternation of assignment and mean that trains each stage.
- N. Zeghidour, A. Luebs, A. Omran, J. Skoglund and M. Tagliasacchi, *SoundStream: an end-to-end neural audio codec*, 2021. A convolutional encoder and decoder with a residual quantizer trained jointly with adversarial losses.
- A. Défossez, J. Copet, G. Synnaeve and Y. Adi, *High fidelity neural audio compression*, 2022. A streaming codec with a residual quantizer and discriminators at several resolutions.
- R. Kumar, P. Seetharaman, A. Luebs, I. Kumar and K. Kumar, *High-fidelity audio compression with improved RVQGAN*, 2023. Periodic activations and codebook-usage fixes in a codec of this family.
- A. Défossez, L. Mazaré and others, *Moshi: a speech-text foundation model for real-time dialogue*, 2024. The codec at 12.5 frames a second with a semantic first codebook, whose design brain's crate follows.
- L. Ziyin, T. Hartwig and M. Ueda, *Neural networks fail to learn periodic functions and how to fix it*, 2020. The periodic activation.

## Exercises

1. **[Calculate]** A codec for 16 kHz audio has strides 2, 4, 5, 8 and 2 in its encoder, 8 codebooks of 1024 entries. Give its hop, its columns per second, its tokens per second and its bits per second.
2. **[Calculate]** For a clip of 5 seconds and 123 samples at 24 kHz, give the number of columns the codec of this chapter produces, the length of the decoded clip and the surplus in milliseconds.
3. **[Calculate]** Three causal layers have $(k, s)$ equal to $(7, 1)$, $(8, 4)$ and $(10, 5)$. Give the receptive field and the jump after each, and the output length for an input of 1000 samples.
4. **[Calculate]** Encode the vector $(0.31, 0.77)$ with the three tables of the hand example, giving the three indices, the sum of the three centres and the squared error after each stage.
5. **[Calculate]** For one SEANet residual unit at 192 channels (a convolution of kernel 7 at dilation 3, then a pointwise convolution, with biases, and two SnakeBeta pairs), count its parameters and its multiply-adds for one second at 8000 positions a second.
6. **[Calculate]** How many floats does the decoder carry for a convolution of kernel 7 and dilation 27 on 100 channels, and for a transposed convolution of kernel 12 and stride 6 to 100 channels?
7. **[Implement]** Reproduce `audio_codecs.token_rates`, `audio_codecs.frame_alignment` and `audio_codecs.encoder_lengths`: the bit arithmetic, the nested ceilings and the receptive-field recursion.
8. **[Implement]** Reproduce `audio_codecs.rvq_trace`, then train the toy of `audio_codecs.rvq_data_check`, `audio_codecs.rvq_stages` and `audio_codecs.rvq_flat` with the book's generator and report the table, saying which cell is most sensitive to the order of your sums.
9. **[Implement]** Reproduce `audio_codecs.codec_counts`, `audio_codecs.attention_work` and `audio_codecs.streaming_state` from the layer list of the chapter.
10. **[Investigate]** With the toy data, compare 3 stages of 4 codes, 2 stages of 8 and one flat table of 64 (all 6 bits) on training and held-out vectors, with the searches each needs. Which is best at fixed bits, and which at fixed search?
11. **[Investigate]** Build the trainer's autoencoder (one convolution from 1 to 4 channels, kernel 3, the 8-entry table of the chapter, one convolution back) on a 48-sample sine, and compute the decoder's weight gradient with the decoder fed the continuous latent and with it fed the quantized latent. How far apart are they, relatively, and which one lowers the loss that the quantized forward reports?
12. **[Investigate]** At what clip length, in columns, does the decoder's window of 72 save at least half of the attention work of an unwindowed decoder? Use the number of keys each query sees.

## Key terms

- **residual vector quantization**: a chain of codebooks in which each quantizes the error left by the ones before it, so that the stored indices are one per stage and the reconstruction is the sum of the chosen entries. <!-- concept: residual-vector-quantization -->
- **neural audio codec**: a trained encoder, quantizer and decoder that turn a waveform into a short sequence of discrete codes and back to a waveform. <!-- concept: neural-audio-codec -->
- **audio token**: one integer of a codec's output, an index into one stage's table for one column, which a language model treats as a symbol. <!-- concept: audio-token -->

## Answers

**Recall questions.**

1. $24000 \times 0.080 = 1920$ samples.
2. It stores the index $k(\mathbf{z})$ of the code at the smallest squared distance, $\log_2 K$ bits; when two codes are equally near, the lower index wins.
3. $r_\ell = r_{\ell-1} + (k-1)j_{\ell-1}$ and $j_\ell = j_{\ell-1}s$; a stride-$s$ convolution with "same" padding on a length $H$ gives $\lceil H/s \rceil$ positions.
4. $(n-1)s + k - 2p$ (plus the output padding), by stamping the kernel at every $s$-th position; where stamps overlap their values add.

**Check your understanding.**

1. 12.5 columns is not a count: a clip of 24,000 samples gives $\lceil 24000/1920 \rceil = 13$ columns and the decoder returns $13 \times 1920 = 24{,}960$ samples, 960 more (40 milliseconds). The caller must keep $n$ and cut the output; only a multiple of 1920 round-trips.
2. The second table is trained on the residuals the first leaves, whose distribution is a different one (smaller and less structured), so its centres sit where the residuals are, not where the data are; scaling the first table would place them in the same pattern as the data.
3. Each code costs 12 bits instead of 11, so the bits per second rise from 2,200 to $12.5 \times 16 \times 12 = 2{,}400$; the search doubles to 65,536 entries and 16,777,216 multiply-adds a column; the number of cells is multiplied by $2^{16} = 65{,}536$.
4. As it stood before commit `6e1a23cdc`, it compared the analytic gradient of the surrogate loss, in which the decoder was fed the continuous latent, with finite differences of the same surrogate. It could not detect that the straight-through estimator should run the decoder on the quantized latent, since that objective is not the one differentiated; this is knowledge #3's lesson.
5. Work is parameters times how many positions they are applied at. Block 4's 333,984 parameters are applied at 24,000 positions a second (and the 192-channel units at 8,000 before it), while the upsampling stages' weights are applied at between 12.5 and 50 positions a second, so a small block at a high rate costs 6,193.2 million and 21 million parameters at a low rate cost 708.3 million.
6. The encoder's attention grows as the square of the clip: tripling the clip multiplies it by about 9, and the table has 2,307,072,000 for 30 seconds against 257,024,000 for 10 seconds, a factor of 8.98. A window of 250 columns would make it grow linearly once the clip passes 10 seconds: the query-key pairs are 31,375 for 250 columns and $31{,}375 + 500 \cdot 250 = 156{,}375$ for 750, a factor of about 5.0, not 9.
7. The device kernel and the host loop were two implementations of one transformer, and the fix went into one. The only test comparing them used 16 columns, below the window, where plain causal attention and windowed attention are the same function, so it could not fail.

**Exercises.**

1. Hop $2 \cdot 4 \cdot 5 \cdot 8 \cdot 2 = 640$ samples, $16000/640 = 25$ columns per second, $25 \times 8 = 200$ tokens per second, $200 \times 10 = 2000$ bits per second.
2. $5 \cdot 24000 + 123 = 120{,}123$ samples; $\lceil 120123/1920 \rceil = 63$ columns; decoded $63 \times 1920 = 120{,}960$ samples, surplus $837$ samples, 34.9 milliseconds.
3. Layer 1: $r = 7$, $j = 1$. Layer 2 ($k = 8$, $s = 4$): $r = 7 + 7 \cdot 1 = 14$, $j = 4$. Layer 3 ($k = 10$, $s = 5$): $r = 14 + 9 \cdot 4 = 50$, $j = 20$. Lengths: 1000, 250, 50.
4. Stage 1: nearest of $(\pm 0.5, \pm 0.5)$ to $(0.31, 0.77)$ is $(0.5, 0.5)$, index 3, residual $(-0.19, 0.27)$, squared error $0.1090$. Stage 2: the corners $(\pm 0.25, \pm 0.25)$: nearest to $(-0.19, 0.27)$ is $(-0.25, 0.25)$, index 2, residual $(0.06, 0.02)$, error $0.0040$. Stage 3: corners $(\pm 0.125, \pm 0.125)$: nearest to $(0.06, 0.02)$ is $(0.125, 0.125)$, index 3, residual $(-0.065, -0.105)$, error $0.01525$. The sum of the three centres is $(0.375, 0.875)$. The error at stage 3 is larger than at stage 2: the grid has no entry at the origin, and a residual smaller than its cell is pushed further off.
5. Parameters: two SnakeBeta pairs $2 \cdot 2 \cdot 192 = 768$; the dilated convolution $192 \cdot 192 \cdot 7 + 192 = 258{,}240$; the pointwise convolution $192 \cdot 192 + 192 = 37{,}056$; total 296,064. Multiply-adds $(192 \cdot 192 \cdot 7 + 192 \cdot 192) \cdot 8000 = 294{,}912 \cdot 8000 = 2{,}359{,}296{,}000$.
6. $100 \cdot 6 \cdot 27 = 16{,}200$ floats for the dilated convolution; $100 \cdot (12 - 6) = 600$ floats for the transposed one.
7. to 9. Match the printed tables to the printed digits. For the trained toy, a different order of the additions in a mean can move the last digit of a cell, never an index, unless two distances tie to rounding; check the stage-1 error first.
10. Run the experiment as the exercise says; the chapter's own numbers fix the flat table at 0.1844 and 0.2357 and the 2 x 8 chain at 0.2248 and 0.2486, so a reader's table must contain those two rows exactly.
11. The two gradients differ because the decoder's input and the gradient at its output both differ; a reader who finds them equal has fed the same latent to both. Which one lowers the loss that the quantized forward reports is what the exercise measures, and the chapter does not claim the result.
12. The unwindowed decoder's work for $T$ columns is proportional to $T(T+1)/2$, the windowed to $\sum_i \min(i+1, 72) = 2628 + 72(T - 72)$ for $T \ge 72$. The windowed work is at most half when $2(2628 + 72(T-72)) \le T(T+1)/2$, which first holds at $T = 246$ (15,156 against 30,381 keys), a clip of 19.7 seconds.
