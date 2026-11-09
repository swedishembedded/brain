<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Positional Information

*Attention ignores order; how do we tell it where each token is?*

Swap the first and last words of "dog bites man" and the layer of the last chapter hands every word exactly the vector it had before. Not nearly the same vector: the same to the last digit the machine keeps. The sentence has changed its meaning completely, any reader knows who is biting whom, and the layer cannot tell. Chapter 7 proved this in general with a permutation matrix and accepted it as the price of parallelism. A recurrent network knew the order because it read in order; this layer reads everything at once, and a thing that is read all at once has no order unless somebody writes it in.

This chapter is about writing it in. There are three places to do it. The position can be added to the vectors that enter the layer, as one more vector per token. It can be applied to the queries and keys, so that the score of a pair depends on where the two tokens are. Or it can be added to the score itself, as a number that depends on how far apart the two tokens are. The first is the oldest, the third is the most direct, and the second is the one that the Qwen decoders in brain use; following it takes most of the chapter, because its consequences reach everywhere: how channels are paired, how far a signal travels, what happens when a model reads past the length it was trained on, and how a picture, which has two directions, is given positions.

Some things from earlier chapters come back first. Answer them from memory; the answers are at the end of the chapter.

1. (Volume V, [Chapter 7](07-self-attention.md), scaled dot-product attention.) The rows of $\mathbf{X}$ are shuffled by a permutation matrix $\boldsymbol{\Pi}$. What happens to $\mathbf{Q}$, $\mathbf{K}$ and $\mathbf{V}$, to the score matrix $\mathbf{S}$, and to the output $\mathbf{O}$?
2. (Volume I, [Chapter 6](../v01-mathematics/06-decompositions.md), rotation matrices.) What is the product $\mathbf{R}(\alpha)\mathbf{R}(\beta)$ of two plane rotations, and what is $\mathbf{R}(\theta)^\top$?
3. (Volume I, [Chapter 14](../v01-mathematics/14-fourier.md), complex numbers.) Compute $(0.6 + 0.8\mathrm{i})\cdot\mathrm{i}$. What does multiplying by $\mathrm{i}$ do to a point of the plane, and what happens to its length?
4. (Volume I, [Chapter 14](../v01-mathematics/14-fourier.md), sinusoids and Euler's formula.) The wave $\cos(0.5\,t)$ advances by 0.5 radians per step. How many steps make one full period, and what is $e^{\mathrm{i}\varphi}$ in terms of cosine and sine?
5. (Volume V, [Chapter 3](03-embeddings.md), the embedding table.) A table has shape $[V, D]$ and a sentence has $T$ token ids. What is the shape of the embedded sentence?
6. (Volume IV, [Chapter 3](../v04-computation/03-roofline.md), arithmetic intensity.) An operation does 3 floating-point operations for every 8 bytes it moves. What is its arithmetic intensity, and on a machine whose ridge point is 10 FLOP per byte, is it limited by arithmetic or by bandwidth?

## The same words in another order

The test needs three words and three small matrices, and the numbers are printed so that every later table can be recomputed. The words are stand-ins for embeddings that a trained model would have; the numbers mean nothing yet.

<!-- worked: positions.sentence_setup -->
Generator: the book's SplitMix64, seed 0x1e0e3e56a1d5e823 (`seed_of` of the name `positions.sentence`). Draws, in this order: the 12 entries of X (one row for each of the words dog, bites, man; four numbers each), then the 8 entries of W_Q, the 8 of W_K and the 8 of W_V (each 4 by 2, row by row), every one `round(normal(), 1)` (`round` to one decimal, ties away from zero). The embeddings are random stand-ins for learned ones; d_k = d_v = 2.

| row of X | word | embedding x (4 numbers) |
|---|---|---|
| 0 | dog | (0.2, -0.6, 1.4, -1.0) |
| 1 | bites | (-0.6, -0.9, 0.2, 0.3) |
| 2 | man | (1.0, 0.4, 0.7, 0.4) |

|  | W_Q (4 by 2) | W_K (4 by 2) | W_V (4 by 2) |
|---|---|---|---|
| row 1 | (-1.4, -1.4) | (-0.4, -0.1) | (0.5, 0.1) |
| row 2 | (-0.1, -0.1) | (-0.2, 1.7) | (0.2, -1.8) |
| row 3 | (-0.2, 0.3) | (-0.2, -0.7) | (0.0, -1.2) |
| row 4 | (0.2, 0.9) | (-1.8, 0.5) | (-1.3, 1.0) |
<!-- /worked -->

The layer is Chapter 7's: queries, keys and values by three projections, scores divided by $\sqrt{d_k}$, a softmax along each row, weighted sums of the values, no mask. It is run twice, on the words in the order dog, bites, man and in the order man, bites, dog. Each word's output is read at the place the word occupies.

<!-- worked: positions.order_test -->
Attention without a mask (the layer of Chapter 7) on the three embeddings of the table above, once in the order dog bites man and once in the order man bites dog.

| word | place in dog bites man | its output | place in man bites dog | its output |
|---|---:|---|---:|---|
| dog | 1 | (0.040, -0.129) | 3 | (0.040, -0.129) |
| bites | 2 | (0.191, -0.595) | 2 | (0.191, -0.595) |
| man | 3 | (-0.161, -0.053) | 1 | (-0.161, -0.053) |

Largest absolute difference between the two outputs of any word: below 1e-12.
<!-- /worked -->

Each word gets the same output in both orders, whatever came before or after it. Nothing in the layer could tell, because nothing in its input differs: a word's vector is its embedding, and the embedding of "dog" does not know whether it is the subject of the sentence. Whatever the model is to learn about order has to reach the layer through its input or through its scores, and for either route something must say what the order is.

What should that something satisfy? A few demands follow from how the layer is used. Different positions must get different signals, or two words in different places would still look alike. The signal of a position must not depend on how long the sequence is, because a key computed when the sequence had five tokens must still be right when it has five hundred (Chapter 12 builds a cache on exactly that). And the model must be able to compare two positions, since what usually matters in language is not that a word is at place 17 but that it is two places after another one. The schemes of this chapter differ in how directly they satisfy the last demand.

## A table of positions to learn

The first scheme copies the one from Chapter 3. A table of **absolute position embeddings** has one row for each position the model will ever see, $\mathbf{W}_{\mathrm{pos}}$ of shape $[T_{\max}, D]$, trained by gradient descent like every other weight. The input at position $i$ is the embedding of its token plus row $i$ of the table:

$$
\mathbf{x}_i = \mathbf{W}_{\mathrm{emb}}[\mathrm{token}_i] + \mathbf{W}_{\mathrm{pos}}[i].
$$

The position is called absolute because each row says "I am place $i$" and nothing else. The layer after it is unchanged. Here a four-row table is added to the words of the example, and the same two orders are run again:

<!-- worked: positions.learned_positions -->
Generator: the book's SplitMix64, seed 0x801e1f19a2a532b2 (`seed_of` of the name `positions.table`). Draws: the 16 entries of the table W_pos (shape [4, 4]), row by row, each `round(0.5 * normal(), 1)`. The input at place a is the embedding of its word plus row a of the table; the projections are those above.

| row of W_pos | values |
|---|---|
| position 0 | (-0.4, 0.0, 0.1, -0.6) |
| position 1 | (-0.1, 1.0, 0.2, 0.6) |
| position 2 | (0.6, 0.6, 0.4, 0.0) |
| position 3 | (-0.5, -0.2, 1.0, 1.0) |

| word | output in dog bites man | output in man bites dog |
|---|---|---|
| dog | (0.665, -1.722) | (-0.234, -1.082) |
| bites | (-0.049, -1.610) | (0.008, -1.265) |
| man | (-0.505, -1.172) | (-0.155, -1.152) |

Largest absolute difference between the two outputs of any word: 0.900. The table has 4 rows, so a sequence of 5 tokens has a token at position 4 and no row to add to it. A table for 4,096 positions of width 1,024 holds 4,194,304 numbers.
<!-- /worked -->

Now each word has a different output in the two orders, because its input is different: "dog" in first place is not the vector "dog" in third place. The table has solved the problem it was built for, and shown its two costs in the last lines. It has $T_{\max}$ rows and no more, so the longest sequence the model can read is fixed on the day the table is built, and a token at position 4 has no row to add. And each row is trained only by the sequences long enough to reach it. The row of position 3,000 gets a gradient only from documents with more than 3,000 tokens, which are few, so the far rows are the worst trained of the table.

A subtler cost sits in the comparison the model must make. Positions 5 and 7 are two apart, and so are positions 105 and 107, but no row of the table says so. The layer sees $\mathbf{x}_i$ and $\mathbf{x}_j$ and must learn, for every pair of positions separately in principle, how their rows relate. The next scheme removes the table and with it the cost of learning that.

## Waves of every wavelength

Vaswani and his co-authors did not learn the positions. They wrote a formula, so that no position needs a trained row and every position, however large, has a vector. The **sinusoidal encoding** of position $i$ in width $D$ has a sine and a cosine for each of $D/2$ rates $\omega_m$:

$$
\mathrm{PE}(i, 2m) = \sin(i\,\omega_m), \qquad
\mathrm{PE}(i, 2m + 1) = \cos(i\,\omega_m),
$$

with $\omega_m = \mathrm{base}^{-2m/D}$ for $m = 0, \dots, D/2 - 1$, where the base is 10,000. The rates fall on a geometric scale, from $\omega_0 = 1$ radian per position down toward $1/10{,}000$. The first pair completes a turn every $2\pi \approx 6.3$ positions and the last in about 63,000, so the pairs behave like the digits of an odometer: the fast pair says where within a few positions, the slow pair says roughly where within the whole sequence, and together they name a position without any two positions getting the same vector.

<!-- worked: positions.sinusoid_table -->
Width D = 8, base 10000. The four rates are w_m = 10000^(-2m/8) (library pow): w0 = 1, w1 = 0.1, w2 = 0.01, w3 = 0.001. Column 2m is sin(i w_m) and column 2m + 1 is cos(i w_m); three decimals.

| position i | sin w0 | cos w0 | sin w1 | cos w1 | sin w2 | cos w2 | sin w3 | cos w3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 |
| 1 | 0.841 | 0.540 | 0.100 | 0.995 | 0.010 | 1.000 | 0.001 | 1.000 |
| 2 | 0.909 | -0.416 | 0.199 | 0.980 | 0.020 | 1.000 | 0.002 | 1.000 |
| 3 | 0.141 | -0.990 | 0.296 | 0.955 | 0.030 | 1.000 | 0.003 | 1.000 |
| 4 | -0.757 | -0.654 | 0.389 | 0.921 | 0.040 | 0.999 | 0.004 | 1.000 |
| 5 | -0.959 | 0.284 | 0.479 | 0.878 | 0.050 | 0.999 | 0.005 | 1.000 |
| 6 | -0.279 | 0.960 | 0.565 | 0.825 | 0.060 | 0.998 | 0.006 | 1.000 |
| 7 | 0.657 | 0.754 | 0.644 | 0.765 | 0.070 | 0.998 | 0.007 | 1.000 |
<!-- /worked -->

![A grid of 32 rows (positions 0 to 31) and 32 columns (channels 0 to 31) of the sinusoidal encoding of width 32 and base 10000, blue for values near +1, white near 0 and orange near -1. The leftmost columns change sign every few rows, the columns further right change slowly, and the rightmost columns alternate between white for the sine and blue for the cosine of an angle that has barely moved.](img/sinusoid-heatmap.svg)

The picture shows the odometer. The first columns flip within a few rows, the middle ones drift over the 32 positions, and the last pairs have not moved visibly: their sine is near 0 and their cosine near 1 at every position shown.

The authors chose the formula for a reason that can be derived. Take one pair, $(\sin(i\omega), \cos(i\omega))$, and ask what it becomes at position $i + k$. By the angle-addition formulas of Volume I,

$$
\begin{aligned}
\sin((i + k)\omega) &= \cos(k\omega)\,\sin(i\omega) + \sin(k\omega)\,\cos(i\omega),\\
\cos((i + k)\omega) &= -\sin(k\omega)\,\sin(i\omega) + \cos(k\omega)\,\cos(i\omega).
\end{aligned}
$$

That is a matrix times the pair at $i$:

$$
\begin{pmatrix}\sin((i + k)\omega)\\ \cos((i + k)\omega)\end{pmatrix}
= \begin{pmatrix}\cos k\omega & \sin k\omega\\ -\sin k\omega & \cos k\omega\end{pmatrix}
\begin{pmatrix}\sin i\omega\\ \cos i\omega\end{pmatrix}.
$$

The matrix is a rotation by $-k\omega$, and it does not contain $i$. For a fixed offset $k$ the encoding at $i + k$ is a fixed linear function of the encoding at $i$, the same for every $i$, and a linear layer can in principle learn it. The block-diagonal matrix of all the pairs does it at once:

<!-- worked: positions.sinusoid_shift -->
Width D = 8, base 10000, shift k = 3. The map applied to each pair (sin(i w_m), cos(i w_m)) is the 2 by 2 matrix with rows (cos(k w_m), sin(k w_m)) and (-sin(k w_m), cos(k w_m)); it does not mention i. First pair of the encoding at i + k, and the same pair obtained from position i by that map, and the dot product of the full rows at i and i + k.

| i | i + k | first pair at i + k (direct) | first pair from i by the map | dot(PE(i), PE(i + k)) |
|---|---:|---:|---:|---:|
| 0 | 3 | (0.1411, -0.9900) | (0.1411, -0.9900) | 1.964890 |
| 2 | 5 | (-0.9589, 0.2837) | (-0.9589, 0.2837) | 1.964890 |
| 10 | 13 | (0.4202, 0.9074) | (0.4202, 0.9074) | 1.964890 |
| 100 | 103 | (0.6230, -0.7822) | (0.6230, -0.7822) | 1.964890 |

Largest absolute difference over all eight entries and the four positions between the mapped row and the direct row: below 1e-12. The four dot products differ by at most below 1e-9.
<!-- /worked -->

The same rotation gives the dot product of two encodings: $\mathrm{PE}(i)\cdot\mathrm{PE}(i + k) = \sum_m \cos(k\omega_m)$, a number that depends on $k$ and not on $i$, as the last column shows. The authors also tried a learned table in place of the formula and reported nearly identical results. What the formula adds is that it can be evaluated at a position longer than any seen in training; whether the model behaves well there is another matter, and the section on extension comes back to it.

There is a catch in how the vector is used, and it is the reason the rest of the chapter exists. The encoding is added to the embedding, and only then multiplied by the projections. For two positions the score is

$$
(\mathbf{x}_i + \mathbf{p}_i)\,\mathbf{W}_Q\mathbf{W}_K^\top\,(\mathbf{x}_j + \mathbf{p}_j)^\top
= \mathbf{x}_i\mathbf{A}\mathbf{x}_j^\top + \mathbf{x}_i\mathbf{A}\mathbf{p}_j^\top
+ \mathbf{p}_i\mathbf{A}\mathbf{x}_j^\top + \mathbf{p}_i\mathbf{A}\mathbf{p}_j^\top,
$$

with $\mathbf{p}_i = \mathrm{PE}(i)$ and $\mathbf{A} = \mathbf{W}_Q\mathbf{W}_K^\top$. Four terms: content with content, content with position twice, and position with position. The identity $\mathrm{PE}(i)\cdot\mathrm{PE}(j)$ depends on the offset, but the score does not contain that dot product; it contains $\mathbf{p}_i\mathbf{A}\mathbf{p}_j^\top$ with a learned $\mathbf{A}$, which depends on the offset only if training makes $\mathbf{A}$ behave, and the position terms are mixed with the content terms in the same sum. The model gets the information, but it must learn to extract it. The natural question is whether the offset could be put into the score by construction instead.

## Rotating the queries and keys

It can, and the tool is one that Volume I derived twice and pointed here for. Take a query $\mathbf{q}$ and a key $\mathbf{k}$ of width $d$ (the width of one head, $d_k$ of Chapter 7) and cut each into $d/2$ pairs of channels. A pair is a point of the plane. Turn the pair $m$ of the query at position $i$ by the angle $i\,\omega_m$ and the pair $m$ of the key at position $j$ by $j\,\omega_m$, and take the dot product of the two turned vectors. This is the **rotary position embedding**, RoPE, of Su and his co-authors.

In matrices, the rotated query is $\tilde{\mathbf{q}}_i = \mathbf{R}_i\mathbf{q}$ with $\mathbf{R}_i$ the block-diagonal matrix of the plane rotations $\mathbf{R}(i\,\omega_m)$ of Chapter 6 of Volume I. Because $\mathbf{R}(\theta)^\top = \mathbf{R}(-\theta)$ and rotations add their angles,

$$
\tilde{\mathbf{q}}_i \cdot \tilde{\mathbf{k}}_j
= \mathbf{q}^\top\mathbf{R}_i^\top\mathbf{R}_j\,\mathbf{k}
= \mathbf{q}^\top\mathbf{R}_{j - i}\,\mathbf{k},
$$

where $\mathbf{R}_{j-i}$ is the same block matrix at the angle $(j - i)\omega_m$. The score depends on the two positions only through $j - i$. Complex numbers say the same thing in one line. Write pair $m$ of the query as the complex number $z_m = q_{2m} + \mathrm{i}\,q_{2m+1}$ and pair $m$ of the key as $u_m$. Turning a pair by an angle $\varphi$ is multiplying by $e^{\mathrm{i}\varphi}$ (Euler's formula, Volume I; the upright $\mathrm{i}$ is the imaginary unit and the italic $i$ a position), and the dot product of two points of the plane is the real part of the conjugate of the first times the second, so

$$
\begin{aligned}
\tilde{\mathbf{q}}_i\cdot\tilde{\mathbf{k}}_j
&= \sum_{m} \mathrm{Re}\!\left(\overline{z_m e^{\mathrm{i}(i\omega_m)}}\;
   u_m e^{\mathrm{i}(j\omega_m)}\right)\\
&= \sum_{m} \mathrm{Re}\!\left(\overline{z_m}\,u_m\,e^{\mathrm{i}(j - i)\omega_m}\right).
\end{aligned}
$$

The two phase factors collapsed into one that holds the offset and nothing else. A pair of tokens at positions $(5, 8)$ and a pair at $(1000, 1003)$ get the same score if their content is the same.

Here is the computation at width 8, with the numbers printed so that it can be done by hand. The rates are the geometric scale of the last section, $\omega_m = \mathrm{base}^{-2m/d}$ with base 10,000 and $d = 8$, computed with the library power function:

<!-- worked: positions.rope_frequencies -->
Generator: the book's SplitMix64, seed 0x3178a93b296b1e70 (`seed_of` of the name `positions.rope`). Draws: the 8 entries of the query q, then the 8 of the key k, each `round(normal(), 1)`. Width d = 8, base 10000: pair m is the channels (2m, 2m + 1) and turns by the angle i * w_m at position i, with w_m = 10000^(-2m/8) (library pow).

q = (-2.4, -0.1, -1.7, 0.5, 1.1, 1.4, 0.6, -0.1), k = (0.5, 0.6, 0.1, -0.1, 3.1, -0.2, -1.2, -1.9)

| pair m | w_m (radians per position) | positions per full turn, 2 pi / w_m |
|---|---:|---:|
| 0 | 1 | 6.283 |
| 1 | 0.1 | 62.83 |
| 2 | 0.01 | 628.3 |
| 3 | 0.001 | 6283 |
<!-- /worked -->

![Eight points on a circle of radius 1, labelled 0 to 7, showing one pair of channels rotated by one radian per position, going around the circle once in about six steps; inside it a circle of radius 0.6 on which eight points for a second pair, rotated by 0.1 radian per position, bunch together near the start.](img/rope-rotation.svg)

The figure shows what the rotation does to a pair: nothing to its length, everything to its direction. The fast pair goes round the circle in about six steps and comes back near where it started, and the slow pair has hardly moved after eight. The same holds for the whole vector.

<!-- worked: positions.rope_vs_position -->
The first pair of q, rotated at positions 0 to 3 (rate w_0 = 1), and the lengths: the pair's own and the whole vector's.

| position i | rotated pair 0 | length of pair 0 | length of the rotated vector |
|---|---:|---:|---:|
| 0 | (-2.4000, -0.1000) | 2.4021 | 3.5285 |
| 1 | (-1.2126, -2.0736) | 2.4021 | 3.5285 |
| 2 | (1.0897, -2.1407) | 2.4021 | 3.5285 |
| 3 | (2.3901, -0.2397) | 2.4021 | 3.5285 |
<!-- /worked -->

The length of the rotated query is the same at every position, as it must be for an orthogonal matrix, so position changes where the query points and not how strong it is. The score test, with offset 3 at five different places and offset $-3$ at two:

<!-- worked: positions.rope_shift -->
The q, k and rates of the last block, adjacent pairs. The score is the dot product of the query rotated to position i with the key rotated to position j.

| query position i | key position j | offset j - i | score |
|---|---:|---:|---:|
| 0 | 3 | 3 | 3.937064 |
| 5 | 8 | 3 | 3.937064 |
| 40 | 43 | 3 | 3.937064 |
| 100 | 103 | 3 | 3.937064 |
| 1000 | 1003 | 3 | 3.937064 |
| 3 | 0 | -3 | 3.334557 |
| 8 | 5 | -3 | 3.334557 |

The first five rows share the offset 3; their largest spread is below 1e-9. For the second row, the rotated query is (-0.7767, 2.2731, -1.7316, -0.3762, 1.0287, 1.4532, 0.6005, -0.0970) and the rotated key is (-0.6664, 0.4074, 0.1414, 0.0021, 3.1061, 0.0484, -1.1848, -1.9095).
<!-- /worked -->

The first five rows are one number to the six decimals printed, and the last two rows are another number: the sign of the offset matters, since $\mathbf{R}_{j-i}$ and $\mathbf{R}_{i-j}$ are different matrices. Any property that depends on the offset only is a **relative position** property, and a scheme whose scores have it is a relative position scheme. RoPE has it by construction, with no table to learn, no new parameter and no addition to the stream of vectors; Shaw, Uszkoreit and Vaswani had earlier built a relative scheme by adding learned vectors, indexed by offset, to the keys inside attention.

Two details of where the rotation goes are worth stating. It is applied to queries and keys, after the projections $\mathbf{W}_Q$ and $\mathbf{W}_K$, and not to values: the output of attention is a weighted sum of values, and rotating those by their absolute position would make the output depend on it. And since a key at position $j$ is rotated by $j\,\omega_m$ once, when it is computed, and never again, a key stored for later queries stays correct however long the sequence grows, which is the second demand of the list above.

The rotation is a small algorithm. In the book's pseudocode, with the two ways of pairing channels that the next section compares:

```text
function rotary(x: array<real>[d], position: int, base: real,
                adjacent: bool) -> array<real>[d]:
    require d mod 2 == 0
    out = zeros(real, [d])
    for m in 0 .. d // 2:
        omega = base ^ (-2 * m / d)         # rate of pair m
        c = cos(position * omega)
        s = sin(position * omega)
        if adjacent:
            a = 2 * m
            b = 2 * m + 1                   # pair m is the channels (2m, 2m+1)
        else:
            a = m
            b = m + d // 2                  # pair m is the channels (m, m+d/2)
        out[a] = x[a] * c - x[b] * s
        out[b] = x[a] * s + x[b] * c
    return out
```

Each pair costs four multiplications and two additions. The frequencies $\omega_m$ are what the sinusoidal encoding already had: the same geometric scale, now acting on the queries and keys instead of being added to the input.

## Two ways to pair the channels

A head has $d$ channels and the rotation needs them in pairs, but nothing says which two channels make a pair. Two conventions are in use. In the **adjacent** one, the pair $m$ is the channels $(2m, 2m + 1)$, the choice of complex numbers read off an array of real numbers two at a time. In the **half-split** one, the pair $m$ is the channels $(m, m + d/2)$: the first half of the head is the real parts and the second half the imaginary parts. Either way the pair $m$ is rotated at the rate $\omega_m$. They are the same operation on different channels, and for a given set of weights they are not interchangeable.

<!-- worked: positions.pairing -->
The q and k of the last blocks and the same rate w_m for pair m in both conventions. Adjacent: pair m is the channels (2m, 2m + 1). Half-split: pair m is the channels (m, m + 4). Scores at three position pairs with the same offset 3; 'adj, half' means the query is rotated with the adjacent convention and the key with the half-split one.

| i | j | adjacent, adjacent | half, half | adj, half | half, adj |
|---|---:|---:|---:|---:|---:|
| 2 | 5 | 3.937064 | -1.948934 | 1.824830 | -8.882897 |
| 10 | 13 | 3.937064 | -1.948934 | 1.720117 | 0.419117 |
| 100 | 103 | 3.937064 | -1.948934 | 6.626390 | 3.384172 |

Spread over the three rows: adjacent, adjacent below 1e-9; half, half below 1e-9; adj, half 4.906; half, adj 12.267. Reordering the channels of q and k so that the half-split pair (m, m + 4) lands at (2m, 2m + 1) and using the adjacent convention gives the half-split scores: largest difference below 1e-9.
<!-- /worked -->

Each convention on its own is a function of the offset: its column is one number in all three rows. The two numbers differ because the same channels are paired with different partners and turned at different rates. The mixed columns, with the query rotated one way and the key the other, are not a function of the offset at all: the score changes with the position of the pair, by a spread of several units on scores of size a few units. The last line shows that the two conventions are one function in two channel orders: reorder the channels of $\mathbf{q}$ and $\mathbf{k}$ so that the pair $(m, m + 4)$ sits at $(2m, 2m + 1)$, and the adjacent rotation reproduces the half-split scores.

That last fact is the bridge to practice. A trained model has learned, through $\mathbf{W}_Q$ and $\mathbf{W}_K$, to put in channels $(2m, 2m + 1)$ or $(m, m + d/2)$ the content that should turn together. Load its weights into an engine with the other convention and no check on a single operation fails: every shape is right and every number is finite. The attention just compares tokens by the wrong angles. The damage also grows with position, since the angle at position 0 is zero in both conventions and the two agree there, so a test on a short prompt may see little. The cure is to implement the convention of the checkpoint, or to permute the rows of the two projection matrices once at load time as above, and to test with a position well past 0.

## How far the signal reaches

The rates $\omega_m$ decide how far apart two tokens can be and still be told apart. Fast pairs separate neighbours and wrap around quickly: they cannot say whether two tokens are 3 or 3 + 6.28 positions apart. Slow pairs cannot separate neighbours, because they hardly move, but they say which part of a long sequence a token is in. Together the pairs name the offset, as the digits of a number name it, provided the slowest pair turns less than once over the longest distance that matters; otherwise the largest distances alias onto small ones.

A computed curve shows what the pairs do to the score. Take a query and a key that agree on every pair (all pairs equal to $1 + 0\mathrm{i}$). Then the score at offset $\Delta = j - i$ is $\sum_m \cos(\Delta\omega_m)$, which is the largest possible, $d/2$, at offset 0. Divided by $d/2$:

<!-- worked: positions.rope_decay -->
Width d = 64 (32 pairs) and every pair of the query and of the key equal to 1 + 0i, so that the score at offset j - i is the sum over the 32 pairs of cos((j - i) w_m); divided by 32, so that offset 0 gives 1. Three decimals; cos is the library cosine.

| offset j - i | base 10^4 | base 10^6 |
|---|---:|---:|
| 0 | 1.000 | 1.000 |
| 1 | 0.966 | 0.975 |
| 2 | 0.884 | 0.914 |
| 4 | 0.748 | 0.825 |
| 8 | 0.700 | 0.788 |
| 16 | 0.605 | 0.708 |
| 32 | 0.611 | 0.710 |
| 64 | 0.435 | 0.648 |
| 128 | 0.284 | 0.462 |
| 256 | 0.354 | 0.587 |
| 512 | 0.185 | 0.423 |
| 1024 | 0.263 | 0.538 |
| 2048 | -0.040 | 0.371 |
| 4096 | 0.003 | 0.259 |
<!-- /worked -->

![Two curves of the rotary score of a query and a key that agree on every pair, divided by its value at distance 0, against distance from 0 to 512 at width 64: for the base 10^4 the curve falls from 1 to about 0.6 by distance 16 and to about 0.2 by 500, with ripples; for the base 10^6 it falls to about 0.7 by 16 and stays between 0.4 and 0.75 over the rest of the range.](img/rope-decay.svg)

The score falls as the distance grows, with ripples. The fast pairs average out within a few positions, the middle pairs within a few hundred, and what remains at long distance is the contribution of the slow pairs, which is why the curve for the base $10^6$ stays high where the curve for $10^4$ has fallen. This is the effect that Su and his co-authors called long-term decay, and the curve is the honest form of it: the decay is a trend, not a monotone law, and the base $10^4$ column of the table is $-0.040$ at distance 2048 and 0.003 at 4096, and it was 0.185 at 512. It also shows why the kernels of the long-context decoders described below declare bases of 1e6 and more. A larger base makes every pair except the first slower, so that at a given distance fewer pairs have wrapped around, and the score for a matching pair of tokens remains distinguishable from noise farther out. The price is on the other side: if all the pairs are slow, near positions are hardly separated, which is why the base is a choice and not a limit.

## Biases that depend only on the offset

The third place to put position is the score. If what the model needs is a preference that depends on the distance, one can add that preference as a number: $s_{ij} + \mathrm{bias}(j - i)$ before the softmax, per head. The bias needs no rotation and no change to the queries and keys; it is an addition to a matrix that already exists, and it is a relative position scheme by its form. Two instances differ in how the number is found.

Raffel and his co-authors, for T5, learn it. A table of numbers, one for each head and each **bucket** (a group of offsets that share one learned number), is indexed by the bucket of the offset: close offsets have a bucket each and far offsets share buckets on a logarithmic scale, so that the number of parameters does not grow with the length. For an encoder, where a query looks both ways, half of the buckets serve keys before the query and half the keys after it:

<!-- worked: positions.relative_buckets -->
32 buckets and a largest distance of 128, so 16 buckets for the keys at or before the query (offset j - i <= 0) and 16 for the keys after it. With n = |j - i|: the bucket is n for n < 8; for n >= 8 it is 8 + floor(log(n / 8) / log(128 / 8) * 8), at most 15 (natural logarithms), and 16 is added when the key is after the query.

| n = abs(j - i) | key behind: j - i = -n | key ahead: j - i = +n |
|---|---:|---:|
| 0 | 0 | 0 |
| 1 | 1 | 17 |
| 3 | 3 | 19 |
| 7 | 7 | 23 |
| 8 | 8 | 24 |
| 9 | 8 | 24 |
| 12 | 9 | 25 |
| 20 | 10 | 26 |
| 40 | 12 | 28 |
| 70 | 14 | 30 |
| 100 | 15 | 31 |
| 127 | 15 | 31 |
| 128 | 15 | 31 |
| 200 | 15 | 31 |
<!-- /worked -->

The eight nearest distances, 0 to 7, have a bucket each in either direction, the distances 8 and 9 share one, the buckets then widen with the distance, and everything from 100 on shares the last bucket of its side. Press, Smith and Lewis, for ALiBi (attention with linear biases), learn nothing: the bias is minus a fixed slope times the distance, and the slopes of the $H$ heads form a geometric sequence, $\mathrm{slope}_h = 2^{-8h/H}$ for $h = 1, \dots, H$, so that some heads care almost only about the last few tokens and others hardly care at all:

<!-- worked: positions.alibi_bias -->
Eight heads: head h has the slope 2^(-8h/8), that is 0.5, 0.25, 0.125, 0.0625, 0.0312, 0.0156, 0.00781, 0.00391. The bias added to the score of query i and key j (j <= i) in head 1 (slope 0.5) is minus the slope times i - j; blank cells are hidden by the causal mask.

| i \ j | 0 | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---:|
| 0 | 0.000 |  |  |  |  |
| 1 | -0.500 | 0.000 |  |  |  |
| 2 | -1.000 | -0.500 | 0.000 |  |  |
| 3 | -1.500 | -1.000 | -0.500 | 0.000 |  |
| 4 | -2.000 | -1.500 | -1.000 | -0.500 | 0.000 |

Weights of query position 5 over the keys 0 to 5 when every content score is equal, so that only the bias decides:

| head h | slope | j = 0 | j = 1 | j = 2 | j = 3 | j = 4 | j = 5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.5 | 0.034 | 0.056 | 0.092 | 0.152 | 0.251 | 0.414 |
| 4 | 0.0625 | 0.142 | 0.151 | 0.161 | 0.171 | 0.182 | 0.194 |
| 8 | 0.00391 | 0.165 | 0.166 | 0.166 | 0.167 | 0.168 | 0.168 |
<!-- /worked -->

The second table is the effect of the bias alone: with every content score equal, head 1 puts 0.414 of the weight on the nearest key and 0.034 on the farthest of the six, and head 8 is nearly uniform. A slope times a distance is linear, so it has no wrap-around and no limit: the bias at distance 10,000 is defined, which is the reason the authors proposed it for reading sequences longer than the ones trained on.

Both schemes cost a full $T \times T$ extra matrix per head, added to scores that were already $T \times T$, and neither can be folded into the queries and keys. That is a cost the rotation does not have, since the rotation of a query and a key is applied once per token. Brain has both: `crates/t5encoder/src/hostbias.rs` builds the table of bucket ids once on the host, in integers, and the device gathers the learned numbers by it, and `crates/genieredux/src/bias.rs` builds the ALiBi slopes and biases that the kernel `attn_scores_causal_bias.wgsl` adds to each score after scaling it. The Qwen decoders use rotary embeddings, and so the rest of the chapter does too.

## Past the trained length

A model is trained on sequences up to some length $L_{\mathrm{train}}$ and then asked to read longer ones. The learned table has nothing to say about position $L_{\mathrm{train}}$: there is no row. A sinusoidal or rotary scheme can compute an angle for any position, and that is the trap. The angle exists; the model has never seen it.

The pairs that matter most are the slow ones. A pair that completes many turns over the trained length has seen every angle many times, and a longer sequence just brings it around again. A pair that completes less than one turn has seen only part of its circle:

<!-- worked: positions.beyond_training -->
Width d = 64 (32 pairs), base 10000, a model trained on positions 0 to 4095. Pair m turns 4096 w_m / (2 pi) times over the trained length.

| pair m | w_m | turns over 4096 positions | less than one turn |
|---|---:|---:|---|
| 0 | 1 | 651.9 | no |
| 8 | 0.1 | 65.19 | no |
| 16 | 0.01 | 6.519 | no |
| 20 | 0.003162 | 2.061 | no |
| 24 | 0.001 | 0.6519 | yes |
| 28 | 0.0003162 | 0.2061 | yes |
| 31 | 0.0001334 | 0.08693 | yes |

Pairs that complete less than one turn over the trained length: 9 of 32 (m = 23 to 31).
<!-- /worked -->

Nine of the 32 pairs, the slowest, have never gone all the way round, so the first position after the trained window sends each of them to an angle that no training example has produced. The model's learned scoring depends on those angles, and the result is not a graceful degradation but an unpredictable one.

The remedy of Chen and his co-authors, **position interpolation**, brings the new positions back into the old range: to read $s$ times the trained length, divide every rate by the factor $s = \mathrm{fac}$, which is the same as dividing every position by it. Position $s L_{\mathrm{train}}$ then gets the angle that $L_{\mathrm{train}}$ had, and no angle is new. The cost is in the fast pairs. Neighbouring tokens that were 1 radian apart on pair 0 are now $1/s$ radians apart, so the finest distinctions the model learned are squeezed. YaRN (yet another RoPE extension), of Peng and his co-authors, keeps what interpolation does for slow pairs and refuses what it does to fast ones. Each pair $m$ turns $n_m = L_{\mathrm{train}}\,\omega_m/2\pi$ times over the trained length. A pair with many turns, at least a bound $n_{\mathrm{fast}}$ (32), keeps its rate; a pair with few, at most $n_{\mathrm{slow}}$ (1), is interpolated; in between, the weight of the interpolated rate rises linearly with the pair index. The index at which a pair has $n$ turns follows from $\omega_m = \mathrm{base}^{-2m/d}$:

$$
c(n) = \frac{d\,\ln\!\big(L_{\mathrm{train}} / (2\pi n)\big)}{2\ln \mathrm{base}},
\qquad
\text{low} = \lfloor c(n_{\mathrm{fast}})\rfloor,\quad
\text{high} = \lceil c(n_{\mathrm{slow}})\rceil,
$$

with low clamped to at least 0 and high to at most $d/2 - 1$, the index of the last pair. With $r_m = \min(1, \max(0, (m - \text{low})/(\text{high} - \text{low})))$, the rate of pair $m$ is $\omega_m' = r_m\,\omega_m/\mathrm{fac} + (1 - r_m)\,\omega_m$. YaRN adds a second correction, an attention-magnitude factor: it multiplies both the cosine and the sine of every pair by $0.1\ln(\mathrm{fac}) + 1$, so that the rotated query and the rotated key are each that much larger and the score is larger by its square. As an algorithm:

```text
function yarn_rates(d: int, base: real, fac: real, trained: int,
                    fast: real, slow: real) -> (array<real>[d // 2], real):
    require d mod 2 == 0 and fac >= 1
    low = max(0, floor(d * log(trained / (fast * 2 * pi)) / (2 * log(base))))
    high = min(d // 2 - 1, ceil(d * log(trained / (slow * 2 * pi)) / (2 * log(base))))
    if low == high:
        high = high + 0.001                 # a step instead of a ramp
    rates = zeros(real, [d // 2])
    for m in 0 .. d // 2:
        omega = base ^ (-2 * m / d)
        r = min(1.0, max(0.0, (m - low) / (high - low)))
        rates[m] = r * omega / fac + (1.0 - r) * omega
    return (rates, 0.1 * log(fac) + 1.0)    # the rates and the factor on cos and sin
```

For width 64, base 10,000, a trained length of 4,096 and a factor of 2, the rates of the three schemes:

<!-- worked: positions.context_extension -->
Width d = 64, base 10000, trained length 4096, extension factor 2 (to 8192). Linear (position interpolation): every rate divided by 2. YaRN with the rotation bounds 32 (fast) and 1 (slow): the correction range is [10, 23], the weight of the interpolated rate rises linearly from 0 at pair 10 to 1 at pair 23, and cos and sin are multiplied by 0.1 ln 2 + 1 = 1.0693147.

| pair m | unscaled w_m | linear: w_m / 2 | YaRN | weight of the interpolated rate |
|---|---:|---:|---:|---:|
| 0 | 1 | 0.5 | 1 | 0.000 |
| 4 | 0.3162 | 0.1581 | 0.3162 | 0.000 |
| 8 | 0.1 | 0.05 | 0.1 | 0.000 |
| 10 | 0.05623 | 0.02812 | 0.05623 | 0.000 |
| 12 | 0.03162 | 0.01581 | 0.02919 | 0.154 |
| 16 | 0.01 | 0.005 | 0.007692 | 0.462 |
| 20 | 0.003162 | 0.001581 | 0.001946 | 0.769 |
| 23 | 0.001334 | 0.0006668 | 0.0006668 | 1.000 |
| 24 | 0.001 | 0.0005 | 0.0005 | 1.000 |
| 27 | 0.0004217 | 0.0002108 | 0.0002108 | 1.000 |
| 31 | 0.0001334 | 6.668e-05 | 6.668e-05 | 1.000 |

Pairs left unchanged by YaRN: 11 of 32; pairs divided by 2 exactly: 9 of 32.
<!-- /worked -->

Pairs 0 to 10 are exactly the original, pairs 23 to 31 are exactly interpolated, and the pairs between are a blend. The effect on the angles that a model would see at the end of the extended window is the next table: where interpolation is needed, at the slow pairs, it brings the angle back to the trained range, and where it is not needed, at the fast pairs, YaRN leaves it alone.

<!-- worked: positions.extension_reach -->
The angle in radians, position times rate, for a few pairs of the table above: at the last trained position (4095), at the last position of the extended window (8191) without any scaling, and at that position with the linear and the YaRN rates.

| pair m | angle at 4095, unscaled | angle at 8191, unscaled | angle at 8191, linear | angle at 8191, YaRN |
|---|---:|---:|---:|---:|
| 0 | 4095 | 8191 | 4096 | 8191 |
| 8 | 409.5 | 819.1 | 409.6 | 819.1 |
| 16 | 40.95 | 81.91 | 40.95 | 63.01 |
| 24 | 4.095 | 8.191 | 4.096 | 4.096 |
| 31 | 0.5461 | 1.092 | 0.5461 | 0.5461 |
<!-- /worked -->

Pair 24 reaches 4.095 radians over the trained window, and at position 8191 it would reach 8.191 unscaled, but 4.096 with either interpolation or YaRN. Pair 0 at the same position is at 8191 radians for YaRN, as it was trained to be, and at 4096 for interpolation, where each step moves it half as far as before. Pair 16, in the transition, is partly squeezed.

These are methods, and brain implements some of them. A larger base is just a different value for the base that every rotary kernel takes as a parameter. `crates/model/src/rope_scaling.rs` reads the declaration of a checkpoint and implements linear scaling (this section's position interpolation), the piecewise scaling that the Llama 3.1 family declares (it keeps the rates of short wavelengths, divides those of long wavelengths by the factor, and blends smoothly in between), YaRN, and an explicit per-channel divisor table; every one of them is turned into one table of rates, and the table-driven kernels (`rope_base_yarn.wgsl`, `rope_paged_yarn.wgsl`) read it and multiply the cosine and sine by the attention factor. A declaration it does not implement, such as one named `dynamic`, is an error and not silently ignored; the history below says why. The YaRN rates are checked against a pinned oracle: the test `pinned_oracle_small_64_factor2` in `crates/model/src/yarn.rs` has the same configuration as the table above, and its last five values begin with the 2.108483e-4 at pair 27 that the table shows to four digits.

## A position for every axis

A picture is not a line. Cut it into patches and the patches form a grid, and a video adds a third axis. The simplest thing to do with a grid is to flatten it into a line, row by row, and give the patches the positions of a line. For a picture of 2 rows and 3 columns that gives the six patches the positions 2 to 7, and a patch is one position from the patch to its left and three from the patch below it. A picture of 2 rows and 5 columns would put the same two patches five apart. The geometric relation, "directly below", has no fixed offset, and a picture of 32 by 32 patches uses 1,024 positions for what is a square of side 32.

The remedy is to give every token one position for each axis, and to let the channels of a head be driven by the position of whichever axis they belong to. The result is **multi-axis rotary embedding**, multi-axis RoPE: a token has a position triple, here (t, h, w) for the frame, the row and the column of a patch, the channel pairs of a head are divided among the three axes, and pair $m$ turns by the position on the axis that owns it times its own rate $\omega_m$.

The triples need bookkeeping, and the example is a prompt of two words, a picture of one frame of 2 rows and 3 columns, and three more words. A word takes the same position on all three axes and advances a counter by one; the picture takes the counter plus its own frame, row and column, and advances the counter by the largest extent of its grid:

```text
c = 0
for each piece of the prompt, in order:
    if the piece is a word:
        append (c, c, c) to positions
        c = c + 1
    else:                                   # a picture of t frames, h rows, w columns
        for a in 0 .. t:
            for b in 0 .. h:
                for e in 0 .. w:
                    append (c + a, c + b, c + e) to positions
        c = c + max(t, h, w)
```

<!-- worked: positions.mrope_positions -->
A prompt of two words, a picture and three words. The picture is cut into 1 frame of 2 rows and 3 columns of patches (6 tokens, in row-major order). A word takes the position (c, c, c) and advances the counter c by 1; the picture takes the positions (c + t, c + h, c + w) for its frame t, row h and column w, and then advances c by max(1, 2, 3) = 3. The counter starts at 0. The index is also the position a one-axis scheme would give the token.

| index | token | (t, h, w) |
|---|---|---:|
| 0 | what | (0, 0, 0) |
| 1 | is | (1, 1, 1) |
| 2 | image patch | (2, 2, 2) |
| 3 | image patch | (2, 2, 3) |
| 4 | image patch | (2, 2, 4) |
| 5 | image patch | (2, 3, 2) |
| 6 | image patch | (2, 3, 3) |
| 7 | image patch | (2, 3, 4) |
| 8 | in | (5, 5, 5) |
| 9 | the | (6, 6, 6) |
| 10 | picture | (7, 7, 7) |
<!-- /worked -->

Three things stand out. The six picture tokens carry only three different values on each of the h and w axes, and a patch and the one below it are now at offset (0, 1, 0) whatever the width of the picture. The word after the picture is at counter 5, and not at 8 as a one-axis count would give: the picture occupied three positions on the largest axis, not six. And the words, which have the same value on all axes, get $(c, c, c)$, which is what plain RoPE at position $c$ would give; that is by design, a test in `crates/qwen3vl/src/mrope.rs` requires it, and it is checked below.

How the pairs are divided among the axes is a choice with a consequence. Divide a head of 8 pairs into 4 for time and 2 each for height and width. If the pairs are cut into blocks, time first, with one scale of rates over the whole head, the height and width axes get only the slow pairs. If the pairs are dealt out in turn, time, height, width, time, height, width, and so on until the smaller sections are used up, every axis gets fast and slow pairs:

<!-- worked: positions.mrope_axes -->
Head width d = 16, so 8 pairs with the rates w_m = 10000^(-2m/16); sections (t, h, w) = (4, 2, 2) pairs. Two ways to assign the pairs to the axes, and the range of rates each axis ends up with (largest to smallest).

| layout | axis of pairs 0 to 7 | rates of t | rates of h | rates of w |
|---|---|---|---|---|
| blocks | t t t t h h w w | 1 to 0.0316 | 0.01 to 0.00316 | 0.001 to 0.000316 |
| interleaved | t h w t h w t t | 1 to 0.000316 | 0.316 to 0.01 | 0.1 to 0.00316 |
<!-- /worked -->

Both exist in brain. The Z-Image port (`crates/dit/src/rope.rs`) uses blocks, of 32, 48 and 48 channels in a head of 128, and avoids the problem the table shows by restarting the scale inside each axis, so that every axis has its own full range of rates from 1 down. The Qwen3-VL builder (`crates/qwen3vl/src/mrope.rs`) keeps one scale over the head and interleaves: its function `axis_map` gives pair $m$ to the height axis when $m \bmod 3 = 1$ and to the width axis when $m \bmod 3 = 2$, for as many pairs as the section sizes allow, and the rest to time; for the sections (24, 20, 20) of a head of width 128 that is 24 time pairs, 20 height and 20 width.

The properties of the single axis carry over, and the check is again numerical. A query and a key of width 16, rotated by their triples, give a score that depends only on the offset triple, and with equal positions on the three axes the rotation is the plain one:

<!-- worked: positions.mrope_offsets -->
Generator: the book's SplitMix64, seed 0x03ca6b0ed89e64f5 (`seed_of` of the name `positions.mrope`). Draws: the 16 entries of q, then the 16 of k, each `round(normal(), 1)`. Head width 16, rates w_m = 10000^(-2m/16), half-split pairs (m, m + 8), interleaved sections (4, 2, 2): pair m turns by (position on its axis) * w_m. The score is the dot product of the rotated query at the first triple with the rotated key at the second.

| query triple | key triple | offset (t, h, w) | score |
|---|---|---|---:|
| (2, 2, 2) | (2, 2, 3) | (0, 0, 1) | -0.734523 |
| (5, 7, 9) | (5, 7, 10) | (0, 0, 1) | -0.734523 |
| (2, 2, 2) | (2, 3, 2) | (0, 1, 0) | -0.535912 |
| (8, 3, 6) | (8, 4, 6) | (0, 1, 0) | -0.535912 |
| (0, 0, 0) | (1, 2, 3) | (1, 2, 3) | -0.445754 |
| (4, 5, 6) | (5, 7, 9) | (1, 2, 3) | -0.445754 |
| (4, 5, 6) | (6, 5, 4) | (2, 0, -2) | -0.338387 |

Rows 1 and 2 share an offset, as do rows 3 and 4, and rows 5 and 6: spreads below 1e-9, below 1e-9 and below 1e-9. With the same position on all three axes the rotated query equals the plain half-split rotation at that position: largest difference below 1e-12 over the positions 0, 3, 17 and 100.
<!-- /worked -->

## What a rotation costs

The rotation is cheap in arithmetic and still not free, and the book's habit is to count. For one token of one layer, take the shapes of the cost section of Chapter 7: width 1,024, 16 heads of width 64, a cache of 4,096 keys. Counting a multiplication or an addition as one operation and the lookup of a cosine or a sine as none:

<!-- worked: positions.rope_cost -->
One token of one layer with D = 1024, H = 16 heads of width d_h = 64 and a cache of T = 4096 keys; a multiplication or an addition is one FLOP; sin, cos and the table lookups are counted apart.

| quantity | FLOPs | rule |
|---|---:|---|
| rotation of q and k | 6,144 | 6 per pair: 4 multiplications, 2 additions |
| scores and weighted sum | 16,777,216 | 4 T d_h per head, T = 4096 |
| four projections | 8,388,608 | 8 D^2 |
| scores and sum / rotation | 2,731 times | ratio of the rows 2 and 1 |
| projections / rotation | 1,365 times | ratio of the rows 3 and 1 |

The rotation reads and writes q and k once each in float32: 16,384 bytes, so it does 0.38 FLOPs per byte. If every head computes its own angles (an analytic kernel), the token costs 3,072 evaluations of pow, sin and cos (each of the 16 heads of the query and each of the 16 heads of the key rotates its 32 pairs, so q and k are counted separately: 2 x 16 x 32 x 3); if the angles come from a table shared by the heads, it costs 64 evaluations of sin and cos, or none when the table is read from memory.
<!-- /worked -->

The attention that follows the rotation does 2,731 times its arithmetic, and the ratio only grows with a longer sequence, since the attention grows with $T$ and the rotation does not; even the four projections do 1,365 times as much. In arithmetic the rotation does not matter. In bytes it does: with 0.38 FLOPs for every byte moved it is far below the ridge points of the machines of Volume IV (10 and 50 FLOP per byte), so it is bandwidth-bound, and the cost of a kernel that rotates $\mathbf{q}$ and $\mathbf{k}$ in memory is the cost of reading and writing them. That is the case for fusion, which Volume IV made in general: do the rotation in the kernel that already holds $\mathbf{q}$ and $\mathbf{k}$ in registers, and the extra read and write disappear. The count of angle evaluations is the other cost: a kernel that computes its own angles in every head evaluates the power, the sine and the cosine once per head and pair, and a table of angles shared by the heads evaluates sines and cosines once per position and pair, or not at all when it is read from memory. The kernels of brain come in both forms, as the next section shows.

## Rotary embeddings in brain

Brain has many rotary kernels, and the reason is the chapter's own. They differ in the convention for pairing, in the part of the head that is rotated, in where the position of a row comes from, and in where the angles are computed. They are the variations of the sections above, each with a checkpoint that needed it.

| what varies | kernels in `crates/kernels/wgsl/` |
|---|---|
| half-split pairs, analytic angles at the row's position in its sequence | `rope_base.wgsl`, `rope_neox.wgsl` (which `rope_partial.wgsl` subsumes at full rotation) |
| half-split, decode: the position is an explicit input | `rope_at.wgsl` (consecutive positions from one base), `rope_paged.wgsl` (one position for each row) |
| half-split, only the first `rot_dim` channels of a head | `rope_partial.wgsl` and `rope_partial_at.wgsl` |
| adjacent pairs, a base given as a parameter | `rope_train.wgsl`, `rope_sub.wgsl` (a sub-slice of the head); `rope.wgsl` has the base 10,000 built in |
| table-driven: the angles come from a host-built table | `rope2d.wgsl`, `rope2d_partial.wgsl` (half-split), `rope_interleave_table.wgsl` (adjacent) |
| with a scaled table and an attention factor | `rope_base_yarn.wgsl`, `rope_paged_yarn.wgsl` |
| fused with the normalization of $\mathbf{q}$ and $\mathbf{k}$ | `qknorm_rope_fused.wgsl` and its siblings |

The convention was a decision from the start. Commit `3bb6a8f77` (June 2026) added `rope_base` and its backward kernel to the Qwen decoder and its message says why a new kernel was needed beside the existing one: the half-split convention was "required for weight-import parity". The header of `crates/kernels/wgsl/rope_base.wgsl` still says it, in the chapter's terms, and contrasts it with `rope_train.wgsl`, which "rotates adjacent interleaved pairs". Moondream, a model that rotates only half of each head, needed another: commit `5922df133` (July 2026) added `rope_partial`, 32 channels of the 64 of a head, with a base of 1.5e6, the other 32 passing through, and its message says that it is new instead of an extension of `rope_base` so that shared kernels stay untouched. The bases in use are also evidence for the section on reach: the header of `rope_base.wgsl` gives 1e6 for the Qwen family, Moondream's is 1.5e6 and GLM-5's 8e6, against 256 for the image grids of Z-Image, which never need positions in the thousands.

The backward pass of a rotary kernel is the transposed rotation: a rotation is orthogonal, its inverse is its transpose, and the transpose of a rotation by $\theta$ is a rotation by $-\theta$, which is how `rope_base_bwd.wgsl` and `rope_train_bwd.wgsl` differ from the forward kernels by a sign of the sine.

Four stories from the history are all stories of position information that was wrong in a way no finiteness test could see.

**A table that was never written.** Every Qwen3-VL decoder rotates through the table-driven `rope2d` kernel, and in commit `160ff7cda` (September 2026) its text-only encoding path was found to read a table that nobody had uploaded. The commit message works out what a zeroed table does, and it is the chapter's own algebra: a zero cosine and a zero sine do not mean no rotation, they give $\mathbf{q}' = \mathbf{q}\cos\theta - \mathbf{q}_{\mathrm{half}}\sin\theta = 0$, so every score is zero, every softmax is a uniform average over the causal prefix, and the output is finite, varies with the tokens, and has no word order. The code confirms the account: `write_text_mrope_tables` in `crates/qwen3vl/src/model.rs` uploads the diagonal positions $(i, i, i)$ and the test `text_only_encode_hidden_applies_positional_rope` compares the batched build with a decode-only build of the same weights, which rotates each token at its own position. The lesson is the one of the start of the chapter: a model without position information does not fail, it reads a bag of tokens.

**A base that two oracles shared.** Commit `e4e41dd79` (September 2026) found the kernels `rope_train`, `rope_train_bwd` and `rope_sub` rotating at a base of 10,000 compiled in, in a model whose configuration declares 8e6. Every test of the model compared one of its paths with another, the cached decode against the full recompute, and both rotated at the same wrong base, so all of them passed (knowledge #177). The remedy is in the test `every_interleaved_rope_kernel_rotates_at_the_callers_theta` of `crates/model/tests/it/interleaved_rope_theta.rs`: it rotates on the host in double precision at the caller's base and compares each kernel with that. The base is now the last field of the kernels' parameters. The kernel `rope.wgsl` still has 10,000 compiled in, the value used by the one model whose configuration declares no base.

**A scaling that was dropped.** Three decoders each read the `rope_scaling` entry of a checkpoint and kept it only when its type was `yarn`; everything else, linear scaling of a code model, the Llama 3.1 scaling and YaRN itself under the spelling `rope_type`, became plain RoPE without an error. Commit `543651959` (September 2026) replaced the three readers by the one of `crates/model/src/rope_scaling.rs` and the refusal of the types it does not implement (knowledge #175). A sibling of this bug had been fixed two weeks earlier in commit `d63b3f031` (September 2026): the table for chunked prefill of a hybrid decoder was the one of three call sites that YaRN never reached, so a model wrote keys at one rotation and read them with queries at another, a defect that, in the message's words, only shows when scaling is on, a round boundary is crossed and a position past the original window is reached, all at once.

**Positions that collided.** Commit `e39f5fb08` (August 2026) fixed an omni model that produced garbage for spoken input: the position builder gave an audio run a position on one axis and pinned the other two at the run's anchor, so every audio row collided on the same height and width coordinates. The reference gives audio a plain diagonal, the same value on all three axes, like text. The fix is the branch of `get_rope_index_multi` for a run with a $1 \times 1$ grid, and the same function documents why the counter advances by `max(t, h, w)` after a run. The decode side has the shifted counter too: a prompt with images has more KV rows than rotary positions, and `PromptRope::at` in the same file gives each generated token its row shifted by the constant that the prompt left over (commit `05f7b182a`, November 2026, "decode past the image at the shifted position").

The stories share one structure. In each, the position signal was present, finite and plausible, and wrong in a way that a test of the shapes, of finiteness or of the model against itself could not see. What guards each now is a comparison against a source that takes the positions from somewhere else: a decode-only build, a host oracle in double precision, a reference implementation.

## Check your understanding

1. A colleague adds sinusoidal vectors to the embeddings and says: "$\mathrm{PE}(i)\cdot\mathrm{PE}(j)$ depends only on $j - i$, so the model sees relative position directly." Find the gap in the argument.
2. A model trained with the half-split pairing is run by a kernel that rotates adjacent pairs. What happens to the score at position 0, what at later positions, and why can a quick test on a short prompt miss it?
3. For width 64, the rate of the slowest pair is $(10^4)^{-62/64}$ at base $10^4$ and $(10^6)^{-62/64}$ at base $10^6$. Which pair does a change of base not move at all, and by about what factor is the slowest pair slowed?
4. Position interpolation by a factor of 2 brings every angle of the extended window back into the trained range. Why can it still hurt a model on short texts?
5. A picture of 2 rows and 3 columns is flattened row by row into one axis. How far apart are a patch and the one below it, how far for a 2 by 5 picture, and what does multi-axis RoPE give in both cases?
6. Why is the rotation applied to queries and keys but not to values?
7. A rotation table that was never written holds zeros. What does attention compute with it, and why does a test that checks only that the output is finite pass?

## Summary

Attention treats its input as a set, so a model must be told where each token is, and there are three places to tell it. A table of learned absolute position embeddings, added to the input, is simple, and ends at the length it was built for. The sinusoidal encoding replaces the table with a formula on a geometric scale of rates, which makes the encoding of a shifted position a fixed linear function of the original, and still adds the position to the content before the projections mix them. The rotary embedding applies the same scale as a rotation of queries and keys after the projections, which makes every score a function of the offset by construction: the derivation was one line with complex numbers and the check a table of equal scores at positions that differ by thousands. Added scores, learned in buckets or fixed as a slope, give the same dependence on the offset at the price of a $T \times T$ matrix per head.

Rotary embeddings bring their own choices. Which channels make a pair is a convention that a checkpoint has baked into its weights, and mixing the two gives scores that are neither convention's. The rates set how far a signal reaches, and the base is how that is tuned. Reading past the trained length takes a frequency table that never asks for an angle the model has not seen, by dividing all rates (interpolation) or only the slow ones (YaRN). A picture needs a position for each axis. The cost of the rotation is a few thousand operations for a token, and its real price is the bytes it moves, so it is fused. Brain carries every one of these variants as a kernel, and its history is a series of cases in which the position signal was wrong and nothing failed.

The model now has everything it needs to look and to know where it is looking. What is missing is the unit that repeats: attention alone does not make a network, and the next chapter puts it between a feed-forward network and the residual connections and normalizations that let dozens of copies be stacked and trained.

## References

- A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser and I. Polosukhin, "Attention is all you need", 2017.
- J. Su et al., "RoFormer: Enhanced transformer with rotary position embedding", 2021 (published 2024).
- P. Shaw, J. Uszkoreit and A. Vaswani, "Self-attention with relative position representations", 2018.
- C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena, Y. Zhou, W. Li and P. J. Liu, "Exploring the limits of transfer learning with a unified text-to-text transformer", 2020.
- O. Press, N. A. Smith and M. Lewis, "Train short, test long: Attention with linear biases enables input length extrapolation", 2022.
- S. Chen, S. Wong, L. Chen and Y. Tian, "Extending context window of large language models via positional interpolation", 2023.
- B. Peng, J. Quesnelle, H. Fan and E. Shippole, "YaRN: Efficient context window extension of large language models", 2023.

## Exercises

1. **[Calculate]** For width 2 and the rate $\omega = 0.5$, write the matrix that maps $\mathrm{PE}(i)$ to $\mathrm{PE}(i + 2)$ and apply it to $\mathrm{PE}(3) = (\sin 1.5, \cos 1.5)$. Compare with $\mathrm{PE}(5)$ to four decimals.
2. **[Calculate]** Width 4, base 100, so the rates are $\omega_0 = 1$ and $\omega_1 = 0.1$; adjacent pairs; $\mathbf{q} = (1, 0, 0, 1)$ and $\mathbf{k} = (1, 0, 1, 0)$. Give the score of the query at position 3 and the key at position 5, pair by pair, and the total. Which positions give the same total?
3. **[Calculate]** For 16 heads, give the ALiBi slopes of the first and last head, and the bias each puts on a key at distance 100.
4. **[Calculate]** Width 128, base 10,000 and a trained length of 2,048. How many of the 64 pairs complete less than one turn over the trained length, and which pair is the first of them?
5. **[Calculate]** Give the FLOPs of the rotation of $\mathbf{q}$ and $\mathbf{k}$ for one token with 32 heads of width 128 (counting 6 for each pair), the FLOPs of the scores and weighted sum against a cache of 8,192 keys ($4 T d_h$ for each head), and how many times more work the attention does.
6. **[Calculate]** A prompt has three words, then a picture of 1 frame, 3 rows and 2 columns, then one word. Give the position triple of every token and the counter after the picture.
7. **[Implement]** Implement the rotation of the chapter in both conventions and reproduce `positions.rope_shift`: the generator and draw order of q and k, the rates, and the scores at the seven position pairs to six decimals.
8. **[Implement]** Reproduce `positions.pairing`: the four score columns at the three position pairs, and the reordering that makes the adjacent rotation reproduce the half-split scores.
9. **[Implement]** Implement the YaRN rates and reproduce the table of `positions.context_extension`: the correction range, the rates of the three schemes for the listed pairs, and the attention factor.
10. **[Implement]** Implement the position triples and the interleaved axis assignment and reproduce `positions.mrope_positions` and `positions.mrope_offsets`: the table of triples, and the scores at the seven pairs of triples.
11. **[Investigate]** Train a small attention model on a task in which the answer is the token that stood three places earlier, once with no position information, once with a learned table, once with the sinusoidal encoding and once with rotary embeddings, and test each at the training length and at twice it. Which of them can learn the task at all, and which still work at twice the length? The chapter did not run this.
12. **[Investigate]** Replace the query and key of `positions.rope_decay` by random vectors drawn from a normal distribution and average the score over many draws at each distance. Does the average show the decay of the table, or only its mean? The chapter did not run this.

## Key terms

- **absolute position embedding**: a table with one trained vector for each position, added to the token's embedding before the first layer, so that the same word at two places enters the network as two different vectors; it cannot name a position beyond its last row. <!-- concept: absolute-position-embedding -->
- **sinusoidal encoding**: a position vector made of sines and cosines at a geometric scale of rates, $\sin(i\omega_m)$ and $\cos(i\omega_m)$, which needs no training and has, for every offset, a fixed linear map from the vector at one position to the vector at another. <!-- concept: sinusoidal-encoding -->
- **rotary position embedding**: a scheme that cuts each query and key into pairs of channels and rotates the pair $m$ of position $i$ by the angle $i\,\omega_m$ after the projections, so that every attention score depends on the two positions only through their offset. <!-- concept: rotary-position-embedding -->
- **relative position**: the offset $j - i$ between a key and a query; a position scheme is relative when the score it produces depends on the two positions only through this offset, whether by rotation, by a learned bias per bucket of offsets or by a fixed slope. <!-- concept: relative-position -->
- **multi-axis rotary embedding**: rotary embedding for tokens that have a position on several axes, such as the frame, row and column of an image patch, in which the channel pairs of a head are divided among the axes and each pair turns by the position on its own axis times its own rate. <!-- concept: multi-axis-rope -->

## Answers

**Recall questions.**

1. $\mathbf{Q}$, $\mathbf{K}$ and $\mathbf{V}$ have their rows permuted the same way, $\boldsymbol{\Pi}\mathbf{Q}$ and so on; the score matrix becomes $\boldsymbol{\Pi}\mathbf{S}\boldsymbol{\Pi}^\top$, the weights likewise, and the output $\boldsymbol{\Pi}\mathbf{O}$. Each position's output is the same vector wherever it moves to.
2. $\mathbf{R}(\alpha)\mathbf{R}(\beta) = \mathbf{R}(\alpha + \beta)$, and $\mathbf{R}(\theta)^\top = \mathbf{R}(-\theta)$, the rotation back.
3. $(0.6 + 0.8\mathrm{i})\cdot\mathrm{i} = 0.6\mathrm{i} + 0.8\mathrm{i}^2 = -0.8 + 0.6\mathrm{i}$: a quarter turn counter-clockwise about the origin; the length, 1, is unchanged.
4. A full period is $2\pi/0.5 = 12.57$ steps. Euler's formula: $e^{\mathrm{i}\varphi} = \cos\varphi + \mathrm{i}\sin\varphi$.
5. $[T, D]$: row $t$ is the table row of token $t$.
6. $3/8 = 0.375$ FLOP per byte, far below the ridge point of 10, so it is bandwidth-bound: its attainable rate is $0.375$ times the bandwidth, and only moving fewer bytes makes it faster.

**Check your understanding.**

1. The score does not contain the dot product $\mathrm{PE}(i)\cdot\mathrm{PE}(j)$. It contains $\mathbf{p}_i\mathbf{A}\mathbf{p}_j^\top$ with $\mathbf{A} = \mathbf{W}_Q\mathbf{W}_K^\top$, which is the offset's function only if training makes $\mathbf{A}$ so, and it comes summed with three terms that involve the content. The offset is available, not built in.
2. At position 0 every angle is 0 and the two conventions agree, so the score of a pair of tokens at position 0 is right. At later positions the wrong channels are rotated by the rates of the pairs, and the error grows with the position; in the example the mixed scores differ from the right ones by amounts of the size of the scores. A prompt of a few tokens has small angles on all but the fastest pairs, so little is wrong, and the failure appears at the lengths where the long-range behaviour matters.
3. Pair 0 has $\omega_0 = \mathrm{base}^0 = 1$ at every base and does not move. The slowest pair is $1.3 \times 10^{-4}$ at base $10^4$ and $1.5 \times 10^{-6}$ at $10^6$, slowed by a factor of about 87.
4. The fastest pair's rate is halved, so neighbouring tokens turn half as far apart as in training, and the finest position distinctions the model learned are squeezed; the angles are inside the trained range but the model reads them at the wrong scale. YaRN leaves the fast pairs alone.
5. A patch and the one below it are 3 apart in the 2 by 3 picture and 5 apart in the 2 by 5 picture, so the same geometric relation has two offsets. Multi-axis RoPE gives (0, 1, 0) in both.
6. The output of attention is a weighted sum of values. If the values were rotated by their absolute position, the output would depend on the absolute positions of the keys and not only on the offsets; the content that is handed over is left alone.
7. A zero cosine and a zero sine rotate $\mathbf{q}$ and $\mathbf{k}$ to zero, so every score is 0 and every softmax is uniform over the causal prefix: the output is an average of values, without any word order. It is finite, it varies with the tokens, and so a finiteness test passes; only a comparison with a build that rotates properly shows the difference.

**Exercises.**

1. $\mathbf{M} = \begin{pmatrix}\cos 1 & \sin 1\\ -\sin 1 & \cos 1\end{pmatrix} = \begin{pmatrix}0.5403 & 0.8415\\ -0.8415 & 0.5403\end{pmatrix}$; applied to $(0.9975, 0.0707)$ it gives $(0.5985, -0.8011)$, which is $(\sin 2.5, \cos 2.5)$ to four decimals.
2. Pair 0: $\mathrm{Re}(\overline{1}\cdot 1\cdot e^{\mathrm{i}\cdot 2\cdot 1}) = \cos 2 = -0.4161$. Pair 1: $z = \mathrm{i}$, $u = 1$, and $\mathrm{Re}(\overline{\mathrm{i}}\cdot 1\cdot e^{\mathrm{i}\cdot 0.2}) = \mathrm{Re}(-\mathrm{i}(\cos 0.2 + \mathrm{i}\sin 0.2)) = \sin 0.2 = 0.1987$. Total $-0.2175$. Every pair of positions with $j - i = 2$ gives it.
3. Slopes $2^{-8h/16} = 2^{-h/2}$: head 1 has $2^{-1/2} = 0.7071$ and head 16 has $2^{-8} = 0.003906$. At distance 100 the biases are $-70.71$ and $-0.3906$.
4. $\omega_m = 10{,}000^{-m/64}$ and one turn needs $2048\,\omega_m \ge 2\pi$, that is $\omega_m \ge 0.003068$, $m \le 40$. So pairs 41 to 63 complete less than one turn: 23 pairs, the first being pair 41.
5. Rotation: $2 \cdot 32 \cdot 64 \cdot 6 = 24{,}576$ FLOPs. Attention: $32 \cdot 4 \cdot 8192 \cdot 128 = 134{,}217{,}728$ FLOPs. The attention does 5,461 times the work of the rotation.
6. Words $(0,0,0)$, $(1,1,1)$, $(2,2,2)$; picture patches $(3,3,3)$, $(3,3,4)$, $(3,4,3)$, $(3,4,4)$, $(3,5,3)$, $(3,5,4)$; the counter after the picture is $3 + \max(1, 3, 2) = 6$, so the last word is $(6,6,6)$.
7. Rates $1, 0.1, 0.01, 0.001$; q = (-2.4, -0.1, -1.7, 0.5, 1.1, 1.4, 0.6, -0.1), k = (0.5, 0.6, 0.1, -0.1, 3.1, -0.2, -1.2, -1.9) from the generator with seed `0x3178a93b296b1e70`; the score at offset 3 is 3.937064 at every position pair, and at offset $-3$ it is 3.334557.
8. At the pairs (2, 5), (10, 13) and (100, 103): adjacent, adjacent 3.937064 and half, half $-1.948934$ at all three; the mixed columns change from row to row. The reordered adjacent rotation matches the half-split scores to below $10^{-9}$.
9. The correction range is [10, 23], the factor on cosine and sine is 1.0693147, 11 of the 32 pairs are unchanged and 9 are exactly halved, and the listed pairs have the rates printed in the table (for instance 0.02919 at pair 12 and 0.007692 at pair 16).
10. The triples are those of the table, ending with the three words at $(5,5,5)$, $(6,6,6)$ and $(7,7,7)$. The pairs of rows 1 and 2 give the same score ($-0.734523$), those of rows 3 and 4 the same ($-0.535912$) and those of rows 5 and 6 the same ($-0.445754$), to the six decimals printed.
11. An investigation. Expect the model without position information to fail at any length, since the answer is a function of order; the learned table to learn it at the training length and fail beyond its last row; the sinusoidal encoding and the rotation to learn it, with rotary embeddings the more likely to survive a longer test, and none of that is guaranteed at twice the length. The chapter did not run it.
12. An investigation. The score of a single draw is not the table's: it adds the cosine terms with the random sign and phase of the product of the two pairs, so the mean over draws of the plain dot product at every offset is zero when the content is independent. The decay shows in the average magnitude or in the score of matching content, not in the mean. The chapter did not run it.
