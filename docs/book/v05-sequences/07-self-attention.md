<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Self-Attention

*How can every position in a sequence look at every other position in parallel?*

Count the steps that separate the first word of a hundred-word paragraph from the last in a recurrent network: ninety-nine, each one rewriting the state that carries the word forward. Count them in the encoder-decoder of the last chapter, between the decoder's question and the digit it needs: one. The decoder asked, the encoder's states answered, and nothing stood between the two but a score and a weighted average. The cure for the single vector was a path of length one, and it worked because the path was short, not because the vector was big.

That answer had a gap in it. The encoder that produced the states was still a chain: its state at the fiftieth digit was reached only by passing through the forty-nine before it, and the decoder still wrote one token after another. The question for this chapter is what is left if the chain is removed and only the look is kept. Every position of one sequence asks a question of every position of the same sequence, and the answers are combined, with no state carried from one position to the next. If that works, all the positions can be computed at once, and any two of them are one step apart.

Attention is assembled from earlier pieces, and the questions below gather them.

1. (Volume V, [Chapter 6](06-seq2seq.md), cross-attention.) In the attention of the last chapter, which vector played the question, which was compared with it to give a score, and which was averaged with the weights?
2. (Volume III, [Chapter 3](../v03-neural-networks/03-softmax.md), the softmax and its loss.) The scores $(0, 0, \ln 3)$ go through a softmax. What are the three weights, and if the third is the correct class, what is the cross-entropy loss and the gradient of the loss with respect to the three scores?
3. (Volume I, [Chapter 4](../v01-mathematics/04-matrices.md), matrix multiplication.) A matrix of shape $[4, 3]$ is multiplied by one of shape $[3, 2]$. What is the shape of the product, and how many multiplications does it take?
4. (Volume III, [Chapter 5](../v03-neural-networks/05-layer-gradients.md), the backward pass of a matrix product.) For $\mathbf{C} = \mathbf{A}\mathbf{B}$, with the gradient $d\mathbf{C}$ of the loss with respect to $\mathbf{C}$ in hand, what are $d\mathbf{A}$ and $d\mathbf{B}$?
5. (Volume I, [Chapter 10](../v01-mathematics/10-probability.md), variance.) Sixteen independent random numbers, each with mean 0 and variance 1, are added. What are the variance and the standard deviation of the sum?
6. (Volume V, [Chapter 5](05-recurrent-networks.md), recurrence.) A recurrent layer reads 100 tokens. How many of its steps must run one after another, and does a device with more processors shorten that number?

## Every token asks, advertises and carries

In the last chapter the decoder's state, turned into a vector by $\mathbf{W}_a$, was the question, each encoder state transformed by $\mathbf{U}_a$ was the label it was matched on, and the state itself was what got averaged. Two sequences played three parts. Self-attention keeps the three parts and lets one sequence play all of them. Every position is at once the one asking, the one that may be found and the one that may be read.

A position needs three different things from its vector, and there is no reason for them to be the same. The question it asks of the others is one thing. The label under which the others find it is another: a token may be easy to find as "a number" without that being what it hands over when found. What it hands over, its content, is a third. A model that is free to choose each of them separately has more to work with than a model that must use one vector for all three, and the cost of the freedom is small, one more matrix. So each position $i$ with input vector $\mathbf{x}_i$ of $D$ numbers gets three vectors, each by its own learned linear map:

$$
\mathbf{q}_i = \mathbf{x}_i\mathbf{W}_Q, \qquad \mathbf{k}_i = \mathbf{x}_i\mathbf{W}_K, \qquad \mathbf{v}_i = \mathbf{x}_i\mathbf{W}_V,
$$

the **query**, the **key** and the **value** of position $i$. With the $T$ inputs stacked as the rows of $\mathbf{X}$, of shape $[T, D]$, one matrix product gives all the queries, $\mathbf{Q} = \mathbf{X}\mathbf{W}_Q$ with $\mathbf{W}_Q$ of shape $[D, d_k]$, and likewise $\mathbf{K}$ with $\mathbf{W}_K$ of shape $[D, d_k]$ and $\mathbf{V}$ with $\mathbf{W}_V$ of shape $[D, d_v]$. Queries and keys share a width $d_k$ because they will be compared by a dot product; the width $d_v$ of the values is free. (The names come from a lookup in a dictionary, with the difference that here the lookup is soft: the query is compared with every key and the answer is a blend of all the values.)

A small case shows every step, and it is the case the rest of the chapter returns to. Four positions, each an input vector of three numbers, queries, keys and values of two numbers each:

<!-- worked: self_attention.example_inputs -->
Generator: the book's SplitMix64, seed 0x5aa27276e1e8e42a (`seed_of` of the name `self_attention.example`). Draws, in this order: the 12 entries of X, row by row, each `round(2 * normal(), 1)`; then the 6 entries of W_Q, the 6 of W_K and the 6 of W_V, each row by row and each `round(normal(), 1)` (`round` to one decimal, ties away from zero). The words are only labels for the four positions; the numbers are random stand-ins for learned embeddings. The width of a query and a key is d_k = 2 and of a value d_v = 2.

| position i | token | x_i (3 numbers) |
|---|---|---|
| 1 | the | (0.1, -1.0, 0.3) |
| 2 | cat | (-0.7, -0.5, 1.2) |
| 3 | sat | (-1.6, 0.1, 1.0) |
| 4 | down | (0.1, -0.6, 0.0) |

|  | W_Q (3 by 2) | W_K (3 by 2) | W_V (3 by 2) |
|---|---|---|---|
| row 1 | (0.8, -1.9) | (0.7, 0.6) | (1.0, -0.4) |
| row 2 | (-0.1, 0.1) | (0.0, 0.2) | (0.1, 0.7) |
| row 3 | (0.1, 1.3) | (1.7, -0.5) | (0.1, 1.3) |
<!-- /worked -->

The numbers mean nothing yet. In a trained model the table of inputs comes from the embedding table of Chapter 3 and the three matrices from training; here they are fixed so that every later number can be recomputed by hand. The projections are three matrix products:

<!-- worked: self_attention.projections -->
Q = X W_Q, K = X W_K, V = X W_V with the X, W_Q, W_K, W_V above. Every entry is a sum of three products of numbers with one decimal, so two decimals are exact: for instance the first entry of q_2 is -0.7 * 0.8 + -0.5 * -0.1 + 1.2 * 0.1 = -0.56 + 0.05 + 0.12 = -0.39.

| i | token | query q_i | key k_i | value v_i |
|---|---|---|---|---|
| 1 | the | (0.21, 0.10) | (0.58, -0.29) | (0.03, -0.35) |
| 2 | cat | (-0.39, 2.84) | (1.55, -1.12) | (-0.63, 1.49) |
| 3 | sat | (-1.19, 4.35) | (0.58, -1.44) | (-1.49, 2.01) |
| 4 | down | (0.14, -0.25) | (0.07, -0.06) | (0.04, -0.46) |
<!-- /worked -->

The three matrices of the table above are different, so the same input gives three different vectors: $\mathbf{x}_2$ yields the query $(-0.39, 2.84)$, the key $(1.55, -1.12)$ and the value $(-0.63, 1.49)$. Which of the three a position uses depends on whether it is asking, being found or being read.

## Scores, weights and a weighted sum

To let position $i$ ask its question of position $j$, compare the query of the first with the key of the second. The comparison of Chapter 6, in its dot-product form, is a number that is large when the two vectors point the same way: $s_{ij} = \mathbf{q}_i\cdot\mathbf{k}_j$. For all pairs at once these are the entries of one matrix product, the score matrix $\mathbf{S} = \mathbf{Q}\mathbf{K}^\top$ of shape $[T, T]$. Row $i$ holds what position $i$ asks of each position, column $j$ what each position asks of $j$. Each score is also divided by $\sqrt{d_k}$, a factor whose reason is the next section's subject; take it for the moment as part of the definition.

<!-- worked: self_attention.score_matrix -->
The score matrix is S = Q K^T: entry (i, j) is the dot product of query q_i with key k_j, so row i holds what position i asks of every position. Four decimals are exact. Then every entry is divided by sqrt(d_k) = sqrt(2) = 0.707107 (to 3 decimals).

| i | token | j = 1 | j = 2 | j = 3 | j = 4 |
|---|---|---:|---:|---:|---:|
| 1 | the | 0.0928 | 0.2135 | -0.0222 | 0.0087 |
| 2 | cat | -1.0498 | -3.7853 | -4.3158 | -0.1977 |
| 3 | sat | -1.9517 | -6.7165 | -6.9542 | -0.3443 |
| 4 | down | 0.1537 | 0.4970 | 0.4412 | 0.0248 |

| i | token | j = 1 | j = 2 | j = 3 | j = 4 |
|---|---|---:|---:|---:|---:|
| 1 | the | 0.066 | 0.151 | -0.016 | 0.006 |
| 2 | cat | -0.742 | -2.677 | -3.052 | -0.140 |
| 3 | sat | -1.380 | -4.749 | -4.917 | -0.243 |
| 4 | down | 0.109 | 0.351 | 0.312 | 0.018 |
<!-- /worked -->

A row of scores must become weights that are positive and add to 1, so that the answer to the question is a blend and not a sum that grows with $T$. That is what the softmax of Volume III does, applied to each row on its own: $p_{ij} = \exp(s_{ij}/\sqrt{d_k}) / \sum_{j'}\exp(s_{ij'}/\sqrt{d_k})$. The answer of position $i$ is then the weighted sum of the values with those weights, $\mathbf{o}_i = \sum_j p_{ij}\mathbf{v}_j$, which for all rows is one more matrix product. In matrix form the whole layer is

$$
\begin{aligned}
\mathbf{P} &= \mathrm{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right),\\
\mathrm{attn}(\mathbf{X}) &= \mathbf{P}\mathbf{V},
\end{aligned}
$$

with the softmax taken along each row: **scaled dot-product attention**, the form of Vaswani and his colleagues (2017). Row 2 of the example, step by step:

<!-- worked: self_attention.weights_and_outputs -->
Row 2 in detail (the softmax subtracts the largest score first, as the stable softmax of Volume III does):

| j | scaled score | minus the largest | exp of that | weight = exp / 1.681 |
|---|---:|---:|---:|---:|
| 1 | -0.742 | -0.603 | 0.547 | 0.326 |
| 2 | -2.677 | -2.537 | 0.079 | 0.047 |
| 3 | -3.052 | -2.912 | 0.054 | 0.032 |
| 4 | -0.140 | 0.000 | 1.000 | 0.595 |

The output of position 2 is the weighted sum of the four value vectors: 0.326 * (0.03, -0.35) + 0.047 * (-0.63, 1.49) + 0.032 * (-1.49, 2.01) + 0.595 * (0.04, -0.46) = (-0.044, -0.252).

Every row, with the sum of its weights and its output o_i = sum_j p_ij v_j (3 decimals):

| i | token | p_i1 | p_i2 | p_i3 | p_i4 | row sum | output o_i |
|---|---|---:|---:|---:|---:|---:|---|
| 1 | the | 0.253 | 0.275 | 0.233 | 0.238 | 1.000 | (-0.504, 0.681) |
| 2 | cat | 0.326 | 0.047 | 0.032 | 0.595 | 1.000 | (-0.044, -0.252) |
| 3 | sat | 0.239 | 0.008 | 0.007 | 0.746 | 1.000 | (0.021, -0.400) |
| 4 | down | 0.227 | 0.289 | 0.278 | 0.207 | 1.000 | (-0.581, 0.814) |
<!-- /worked -->

![The data flow of one attention head on the four-token example. The input X, 4 by 3, goes through three separate learned products to give Q, K and V, each 4 by 2. Q and K meet in the product Q K transposed divided by the square root of d_k, giving the 4 by 4 matrix S; an optional mask adds minus infinity to hidden entries before the softmax of each row gives P; P multiplies V to give the output O, 4 by 2.](img/qkv-data-flow.svg)

The figure shows the same flow with the shapes of the example. Position 2 puts 0.595 of its weight on position 4 and 0.326 on position 1, almost none on 2 and 3, and its output is a blend dominated by those two values. Position 1 spreads its weight almost evenly, 0.253, 0.275, 0.233 and 0.238. The reason is in the queries: $\mathbf{q}_1 = (0.21, 0.10)$ is short, so its dot product with every key is near zero and the softmax of near-equal scores is near uniform. A short query asks nothing in particular; a long one that points at a key asks for it.

The same computation, as pseudocode with the optional mask of a later section already in place:

```text
function attention(X: array<real>[T, D], Wq: array<real>[D, dk],
                   Wk: array<real>[D, dk], Wv: array<real>[D, dv],
                   causal: bool) -> (array<real>[T, dv], array<real>[T, T]):
    Q = matmul(X, Wq)                       # [T, dk], row i is the query of i
    K = matmul(X, Wk)                       # [T, dk]
    V = matmul(X, Wv)                       # [T, dv]
    S = matmul(Q, transpose(K))             # [T, T], S[i, j] = dot(Q[i, :], K[j, :])
    P = zeros(real, [T, T])
    for i in 0 .. T:
        row = [S[i, j] / sqrt(dk) for j in 0 .. T]
        if causal:
            for j in i + 1 .. T:
                row[j] = -infinity          # hide the positions after i
        P[i, :] = softmax(row)              # the weights of row i add to 1
    return (matmul(P, V), P)
```

Nothing in this function has a parameter that depends on $T$. The same three matrices serve a sequence of 4 positions or of 4,000, which a recurrent network's state also allowed, and unlike the recurrent network, the work for position $i$ does not wait for position $i - 1$.

The attention of Chapter 6 is this function with one change. Its queries came from the decoder's states and its keys and values from the encoder's: the call is `attention` with $\mathbf{Q} = \mathbf{X}_{\mathrm{dec}}\mathbf{W}_Q$ and $\mathbf{K}, \mathbf{V}$ computed from $\mathbf{X}_{\mathrm{enc}}$, and the score matrix is then $[T_{\mathrm{dec}}, T_{\mathrm{enc}}]$ instead of square. Cross-attention is the general form, in which the queries and the keys and values are free to come from different sequences, and self-attention is the case in which they come from the same one. The scores of Chapter 6 were additive, a small network on the two vectors; the dot product has no parameters of its own and costs $2 d_k$ FLOPs per pair against a layer's worth, which is much of why it became the form in use.

## Why the square root of the width

The division by $\sqrt{d_k}$ is not decoration, and a short derivation shows what it fixes. Suppose a query and a key each have $d_k$ entries that are independent, with mean 0 and variance 1. (This is roughly what unit-variance inputs and the variance-preserving initialization of Volume III give: each entry of $\mathbf{q}$ is a sum of $D$ products with weights of variance $1/D$.) The dot product is a sum of $d_k$ terms $q_l k_l$. Each term has mean $\mathbb{E}[q_l]\,\mathbb{E}[k_l] = 0$, and its variance is $\mathbb{E}[q_l^2 k_l^2] - 0 = \mathbb{E}[q_l^2]\,\mathbb{E}[k_l^2] = 1$, by independence. The $d_k$ terms are independent, and the variance of a sum of independent numbers is the sum of their variances, so

$$
\begin{aligned}
\mathrm{Var}[\mathbf{q}\cdot\mathbf{k}] &= d_k, \qquad \text{standard deviation } \sqrt{d_k},\\
\mathrm{Var}\!\left[\frac{\mathbf{q}\cdot\mathbf{k}}{\sqrt{d_k}}\right] &= 1.
\end{aligned}
$$

Scores grow in size with the width of the vectors that make them, and dividing by $\sqrt{d_k}$ gives them unit variance whatever the width. A sample check, with generator draws and widths printed:

<!-- worked: self_attention.scale_variance -->
Generator: the book's SplitMix64, seed 0x59b936a2d025c35a (`seed_of` of the name `self_attention.variance`). For each width d_k in the table, 2,000 pairs in this order: d_k draws of `normal()` for the entries of q, then d_k for the entries of k; the dot product of each pair is formed. The variance is the sample variance with divisor n - 1. Tolerance: the sample variance of 2,000 values has a relative standard error of about 0.032 for wide vectors and 0.063 for d_k = 1, where the product of two normal numbers has heavier tails.

| d_k | predicted standard deviation sqrt(d_k) | sample variance of q . k | sample standard deviation | sample variance of (q . k) / sqrt(d_k) |
|---|---:|---:|---:|---:|
| 1 | 1.000 | 0.92 | 0.962 | 0.925 |
| 4 | 2.000 | 4.02 | 2.004 | 1.004 |
| 16 | 4.000 | 15.85 | 3.981 | 0.991 |
| 64 | 8.000 | 66.13 | 8.132 | 1.033 |
| 256 | 16.000 | 265.37 | 16.290 | 1.037 |
<!-- /worked -->

At $d_k = 256$ the sample standard deviation is 16.290 against a prediction of 16, and after the division the sample variance is 1.037 (within the stated tolerance of 1). Why does a spread of 16 matter? Because the softmax exponentiates it. Two keys whose scores differ by one standard deviation get a weight ratio of $e^{16}$, about nine million to one, and the row is no longer a blend: one weight is close to 1 and the others are close to 0. The next experiment measures that for rows of 8 keys.

<!-- worked: self_attention.softmax_saturation -->
Generator: the book's SplitMix64, seed 0x15010492a2687dff (`seed_of` of the name `self_attention.saturation`). For each width d_k, 500 rows in this order: d_k draws of `normal()` for the query, then 8 keys of d_k draws each. A row is softmax of the 8 dot products, once as they are and once divided by sqrt(d_k). Entries: the mean over the rows of the largest weight; the mean of 1 - sum of squared weights (the sum of the diagonal slopes p_i (1 - p_i) of the softmax, which is how much gradient the row can pass on); and the number of rows whose largest weight exceeds 0.99. Eight equal weights would be 0.125 and 0.875.

| d_k | largest weight, unscaled | gradient signal, unscaled | rows above 0.99, unscaled | largest weight, scaled | gradient signal, scaled | rows above 0.99, scaled |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.319 | 0.776 | 1/500 | 0.319 | 0.776 | 1/500 |
| 4 | 0.542 | 0.587 | 9/500 | 0.354 | 0.765 | 0/500 |
| 16 | 0.751 | 0.342 | 52/500 | 0.362 | 0.765 | 0/500 |
| 64 | 0.883 | 0.166 | 181/500 | 0.363 | 0.767 | 0/500 |
| 256 | 0.919 | 0.114 | 290/500 | 0.355 | 0.770 | 0/500 |
<!-- /worked -->

Without the scale, the mean largest weight climbs from 0.319 at $d_k = 1$ to 0.919 at $d_k = 256$, and 290 of the 500 rows put more than 0.99 of their weight on one key. With the scale it stays between 0.32 and 0.36 at every width, and from $d_k = 4$ on no row exceeds 0.99. The second column of each half is what the training cares about. The slope of a softmax output with respect to its own score is $p_i(1 - p_i)$, so the sum of the slopes, $1 - \sum_i p_i^2$, measures how much gradient the row can pass back. It falls from 0.776 at $d_k = 1$ to 0.114 at $d_k = 256$ without the scale and stays near 0.77 with it. A saturated row is a row that has stopped learning: its gradient is close to 0 before training has begun, at the point where the weights are random and no weight deserves that much confidence.

None of this makes a hard lookup wrong. A trained model may well put a weight of 0.99 on one key, and the lookup experiment at the end of this chapter ends there. The scale decides where training starts, not where it may finish. Models that use other scales exist, and brain has a kernel with a configurable one (the last section says where); what they share is that the scale is chosen once, in exactly one place, which later turns out to matter for testing.

## A set of positions, not a sequence

Look at what the computation did with the order of the positions. It used none. Query $i$ depends on $\mathbf{x}_i$ alone, and the output $\mathbf{o}_i = \sum_j p_{ij}\mathbf{v}_j$ is a sum over all $j$, which has no first term and no last. Shuffle the rows of $\mathbf{X}$ and every query, key and value is shuffled the same way, the score matrix has its rows and columns shuffled by the same permutation, the softmax of a shuffled row is the shuffled softmax, and the sum is the same sum in another order. In symbols, for a permutation matrix $\boldsymbol{\Pi}$ with $\boldsymbol{\Pi}^\top\boldsymbol{\Pi} = \mathbf{I}$:

$$
\begin{aligned}
\mathbf{Q}, \mathbf{K}, \mathbf{V} &\to \boldsymbol{\Pi}\mathbf{Q}, \boldsymbol{\Pi}\mathbf{K}, \boldsymbol{\Pi}\mathbf{V},\\
\mathbf{S} &\to \boldsymbol{\Pi}\mathbf{S}\boldsymbol{\Pi}^\top,\\
\mathbf{P} &\to \boldsymbol{\Pi}\mathbf{P}\boldsymbol{\Pi}^\top,\\
\mathbf{O} &\to \boldsymbol{\Pi}\mathbf{P}\boldsymbol{\Pi}^\top\boldsymbol{\Pi}\mathbf{V} = \boldsymbol{\Pi}\mathbf{O}.
\end{aligned}
$$

The outputs are shuffled exactly as the inputs were: attention is **equivariant** to permutations of the positions: permuting the inputs permutes the outputs in the same way, where an invariant function would leave the outputs unchanged. The example shows it:

<!-- worked: self_attention.equivariance -->
Generator: the book's SplitMix64, seed 0xe2b3e4a6655e44d3 (`seed_of` of the name `self_attention.shuffle`). The permutation is `shuffle([0, 1, 2, 3], rng)` (Fisher-Yates as in Volume I): [1, 3, 2, 0], so position a of the shuffled input holds token perm[a] + 1 of the original. The example above, attention without a mask, run on the original and on the shuffled input.

| position a of the shuffled input | its token | perm[a] + 1 | output at a, shuffled input | output at perm[a] + 1, original input |
|---|---|---|---|---|
| 1 | cat | 2 | (-0.044, -0.252) | (-0.044, -0.252) |
| 2 | down | 4 | (-0.581, 0.814) | (-0.581, 0.814) |
| 3 | sat | 3 | (0.021, -0.400) | (0.021, -0.400) |
| 4 | the | 1 | (-0.504, 0.681) | (-0.504, 0.681) |

Largest absolute difference between the two output columns: below 1e-12. With the causal mask of the next section the same comparison gives a largest difference of 1.608.
<!-- /worked -->

The two columns agree, to round-off. A layer of this kind treats its input as a set of vectors with no order. The model cannot tell "the dog bit the man" from "the man bit the dog" if the only things it has are the embeddings of the words, which is an odd property for a model of language, and the reason that Chapter 8 is about telling the model where each token is. The recurrent network had order built into the way it read; this layer gave it up in exchange for the parallelism, and the price is paid there.

The last sentence under the table is a hint about the next section: when a causal mask hides part of each row, the same shuffle gives a difference of 1.608, not round-off. A mask makes each position see a different set, so the outputs are no longer a relabelling of each other. That asymmetry is real, and it is not a replacement for knowing positions.

## Hiding the future

A language model of Chapter 4 reads a whole sequence at once during training and must be penalised for its prediction at position $t$ of the token at position $t + 1$. The data takes care of half of this, with targets shifted by one. The other half is the architecture: at position $t$ the layer above has the whole sequence in front of it, and a score matrix that lets query $t$ look at key $t + 1$ lets the model copy its answer. Chapter 4 gave a test that applies to any architecture: replace one input token and see which outputs change; in a causal model those before it cannot. The recurrent network passed it by the order in which it read. Attention passes it by hiding the keys that come later.

The hiding is done to the scores before the softmax. Add to $\mathbf{S}/\sqrt{d_k}$ a **causal mask** $\mathbf{M}$ with $M_{ij} = 0$ for $j \le i$ and $M_{ij} = -\infty$ for $j > i$: the entries above the diagonal become $-\infty$. The softmax gives $\exp(-\infty) = 0$ to those entries, and the weights of row $i$ are spread over the positions $1, \dots, i$ only. The diagonal entry $j = i$ is never hidden, so every row has at least one finite score, the largest score subtracted by the stable softmax is finite, and no row is $0/0$. Row 1 sees only itself and its weight is exactly 1.

<!-- worked: self_attention.causal_weights -->
The scaled scores of the example, with the entries above the diagonal replaced by -infinity before the softmax:

| i | token | p_i1 | p_i2 | p_i3 | p_i4 | output o_i |
|---|---|---:|---:|---:|---:|---|
| 1 | the | 1.000 | 0.000 | 0.000 | 0.000 | (0.030, -0.350) |
| 2 | cat | 0.874 | 0.126 | 0.000 | 0.000 | (-0.053, -0.118) |
| 3 | sat | 0.940 | 0.032 | 0.027 | 0.000 | (-0.033, -0.226) |
| 4 | down | 0.227 | 0.289 | 0.278 | 0.207 | (-0.581, 0.814) |
<!-- /worked -->

![Attention weights of the four-token example as two grids of shaded cells, rows the querying position and columns the position looked at, each cell labelled with its weight. Left, no mask: all sixteen cells have weight. Right, the causal mask: the six cells above the diagonal are marked minus infinity and the remaining weights in each row add to 1; the last row is the same as without the mask.](img/attention-heatmaps.svg)

The first row of the masked table has $p_{11} = 1.000$ and so $\mathbf{o}_1 = \mathbf{v}_1 = (0.030, -0.350)$, exactly the first value. The last row is the same as without the mask, since position 4 sees everyone either way. The others have lost the weight they gave to later positions and the softmax has handed it to the earlier ones: position 2 had 0.326 on position 1 and now has 0.874.

A mask has an edge, and edges are where such code goes wrong. The correct boundary and two wrong ones on the same example, with no mask at all for comparison, each tested by the perturbation test of Chapter 4: replace token 3 and list the positions whose output changes.

<!-- worked: self_attention.mask_boundary -->
The input of the example with the first entry of x_3 increased by 1 (token 3 is replaced). The table lists the positions whose output changes by more than 1e-12 when it is, for four versions of the boundary of the mask; a position whose output is nan (no score survives the mask) is named in brackets and left out of the comparison. A causal model must not change at positions 1 and 2, which come before the token that was replaced.

| mask (the pair i, j is kept when) | positions whose output changes | problem |
|---|---|---|
| no mask | 1, 2, 3, 4 | - |
| causal: j <= i | 3, 4 | - |
| one too tight: j < i | 3, 4 (position 1: nan) | output undefined (nan) at position 1 |
| one too loose: j <= i + 1 | 2, 3, 4 | - |
<!-- /worked -->

The correct mask changes only positions 3 and 4, the one that holds the changed token and the one after it. One step too loose, so that position $i$ also sees $i + 1$, lets the change reach position 2, which comes before it: a leak, silent, because shapes, losses and gradients are all fine. One step too tight, hiding the diagonal as well, leaves position 1 with no score to normalise, a row of $-\infty$ whose softmax divides $0$ by $0$, and the output is not a number; this failure is at least loud. The test, read at the positions before the replaced token, finds both. Brain once trained a model with the leak, and the last section tells how it was found.

The mask has a second job, and it is a large part of the reason this layer replaced the recurrence. A sequence of $T$ tokens gives $T$ next-token problems, one for each position, each of which must see only its own prefix. The mask lets all of them be solved in one pass: position $t$'s output is a function of $x_1, \dots, x_t$ only, so the loss may be the mean of the $T$ per-position cross-entropies of Chapter 4, computed together from one $[T, T]$ matrix. How that compares with the recurrence in work and in depth, counted for the sizes of Chapter 5:

<!-- worked: self_attention.depth_and_work -->
An LSTM layer of 1024 units on an input of width 1024 (8 H (H + D) + 17 H = 16,794,624 FLOPs per step, as in Chapter 5) against one attention layer of width 1024 with 16 heads. Sequential stages: the longest chain of operations in which each one needs the result of the one before it, five for attention (the projections, the scores, the softmax, the average, the output projection) whatever T is. Path: the number of operations between the input at position 1 and the output at position T.

| T | LSTM FLOPs | LSTM sequential stages | attention layer FLOPs | attention sequential stages | LSTM path, position 1 to T | attention path |
|---|---:|---:|---:|---:|---:|---:|
| 16 | 268,713,984 | 16 | 135,286,784 | 5 | 15 | 1 |
| 256 | 4,299,423,744 | 256 | 2,421,161,984 | 5 | 255 | 1 |
| 4,096 | 68,790,779,904 | 4,096 | 104,421,392,384 | 5 | 4,095 | 1 |
<!-- /worked -->

At $T = 4096$ the attention layer does more arithmetic than the LSTM, 104,421,392,384 FLOPs against 68,790,779,904, and it needs five sequential stages where the LSTM needs 4,096. The arithmetic of attention is wider, not smaller, and every product in it can use every processor of the device at once, where the recurrent chain left most of them idle. A path from position 1 to position $T$ in the recurrence goes through 4,095 steps and each one rewrites the state; in attention it is one product. That is what the question of this chapter was asking for. (Generating text is still one token after another, since each new token must exist before the next is asked for; Chapter 12 returns to what is saved there.)

## Several heads, one width

One attention layer gives each position one weighted average, built from one set of weights. A position that needs two different things from two different positions has to get both through the same row. Take tokens that have two properties, a shape and a colour, and suppose a position wants the colours of the tokens that share its shape and, separately, the shapes of the tokens that share its colour. The two requests ask different questions of the same set, and one blend of weights must serve both.

**Multi-head attention** gives each request its own attention. Split the width $D$ into $H$ **heads** of width $d_h = D/H$. Head $h$ has its own three matrices, each of shape $[D, d_h]$, and runs the attention of the last sections on them; the $H$ outputs, each of width $d_h$, are placed side by side to give a $[T, D]$ matrix, which an output matrix $\mathbf{W}_O$ of shape $[D, D]$ mixes:

$$
\begin{aligned}
\mathbf{O}_h &= \mathrm{softmax}\!\left(\frac{\mathbf{Q}_h\mathbf{K}_h^\top}{\sqrt{d_h}}\right)\mathbf{V}_h,\\
\mathbf{Y} &= [\,\mathbf{O}_1 \;\; \mathbf{O}_2 \;\cdots\; \mathbf{O}_H\,]\,\mathbf{W}_O .
\end{aligned}
$$

The division is by $\sqrt{d_h}$, the width of the vectors that are compared. Written this way the heads look like $H$ separate layers, and it is cheaper to see them as one. Put the $H$ matrices $\mathbf{W}_Q^{(h)}$ side by side into one matrix of shape $[D, D]$: one product $\mathbf{X}\mathbf{W}_Q$ gives every query of every head, head $h$ owning columns $h\,d_h$ up to $(h+1)\,d_h$ of the result. A head is a slice of columns, and the concatenation is free: each head writes into its own columns of the output and nothing is moved.

```text
function multi_head(X: array<real>[T, D], Wq: array<real>[D, D],
                    Wk: array<real>[D, D], Wv: array<real>[D, D],
                    Wo: array<real>[D, D], H: int, causal: bool)
        -> array<real>[T, D]:
    require D mod H == 0
    dh = D // H
    O = zeros(real, [T, D])
    for h in 0 .. H:
        a = h * dh
        b = a + dh                          # head h owns columns a .. b - 1
        (Oh, Ph) = attention(X, Wq[:, a:b], Wk[:, a:b], Wv[:, a:b], causal)
        O[:, a:b] = Oh                      # concatenation = writing side by side
    return matmul(O, Wo)
```

A constructed example makes the gain visible. Six features, three for a shape (A, B, C) and three for a colour (red, blue, green), one-hot in each group; four tokens A-red, B-red, A-blue and C-blue. Head 1 is built to compare shapes (its queries and keys read only the shape features) and to carry the colour features as values; head 2 compares colours and carries shapes. These matrices are written by hand here; training would find them, if it found anything like them.

<!-- worked: self_attention.heads_weights -->
Four tokens with six features: three for the shape (A, B, C) and three for the colour (red, blue, green), each a one-hot group. The tokens are 1 = A-red, 2 = B-red, 3 = A-blue, 4 = C-blue. Head 1 has W_Q = W_K = 2 on the three shape features (zero on the colour features) and its values are the colour features; head 2 does the same with the roles swapped; both have d_k = d_v = 3, so scores are divided by sqrt(3) = 1.732; W_O is the identity. The single head for comparison has width 6, W_Q = W_K = 2 on all six features and values equal to the inputs. No mask. Weights to 3 decimals, one row for each position i:

Head 1 (matches on shape):

| i | token | j = 1 | j = 2 | j = 3 | j = 4 |
|---|---|---:|---:|---:|---:|
| 1 | A-red | 0.455 | 0.045 | 0.455 | 0.045 |
| 2 | B-red | 0.077 | 0.770 | 0.077 | 0.077 |
| 3 | A-blue | 0.455 | 0.045 | 0.455 | 0.045 |
| 4 | C-blue | 0.077 | 0.077 | 0.077 | 0.770 |

Head 2 (matches on colour):

| i | token | j = 1 | j = 2 | j = 3 | j = 4 |
|---|---|---:|---:|---:|---:|
| 1 | A-red | 0.455 | 0.455 | 0.045 | 0.045 |
| 2 | B-red | 0.455 | 0.455 | 0.045 | 0.045 |
| 3 | A-blue | 0.045 | 0.045 | 0.455 | 0.455 |
| 4 | C-blue | 0.045 | 0.045 | 0.455 | 0.455 |

One head on all six features:

| i | token | j = 1 | j = 2 | j = 3 | j = 4 |
|---|---|---:|---:|---:|---:|
| 1 | A-red | 0.700 | 0.137 | 0.137 | 0.027 |
| 2 | B-red | 0.154 | 0.786 | 0.030 | 0.030 |
| 3 | A-blue | 0.137 | 0.027 | 0.700 | 0.137 |
| 4 | C-blue | 0.030 | 0.030 | 0.154 | 0.786 |
<!-- /worked -->

The first two grids are the two requests answered separately. In head 1 tokens 1 and 3, both shape A, look at each other (0.455 each, the rest 0.045); in head 2 tokens 1 and 2, both red, do. The single head of width 6 compares all six features at once, and token 1 can only put 0.700 on itself and 0.137 on each of the two tokens that share one property with it. It cannot say "the same shape, from the same shape's point of view" and "the same colour, from the same colour's point of view" in two separate rows.

<!-- worked: self_attention.heads_output -->
Outputs of the two heads concatenated (the first three numbers are head 1's, the last three head 2's), and of the single head of width 6. The first group of the two-head output is a weighted mean of the colour features, the second a weighted mean of the shape features:

| i | token | head 1: colours of same-shape tokens | head 2: shapes of same-colour tokens | one head: all six features at once |
|---|---|---|---|---|
| 1 | A-red | (0.500, 0.500, 0.000) | (0.500, 0.455, 0.045) | (0.837, 0.137, 0.027, 0.837, 0.163, 0.000) |
| 2 | B-red | (0.847, 0.153, 0.000) | (0.500, 0.455, 0.045) | (0.184, 0.786, 0.030, 0.940, 0.060, 0.000) |
| 3 | A-blue | (0.500, 0.500, 0.000) | (0.500, 0.045, 0.455) | (0.837, 0.027, 0.137, 0.163, 0.837, 0.000) |
| 4 | C-blue | (0.153, 0.847, 0.000) | (0.500, 0.045, 0.455) | (0.184, 0.030, 0.786, 0.060, 0.940, 0.000) |
<!-- /worked -->

What the two heads return is what was asked: for token 1 (A-red) the colours of the tokens that share its shape, an even mix of red and blue, $(0.500, 0.500, 0.000)$, and the shapes of the tokens that share its colour, $(0.500, 0.455, 0.045)$. The single head's output is one row that mixes all of it.

What do heads cost? Less than one might think.

<!-- worked: self_attention.head_parameters -->
The width D = 6 split into H heads of width D / H, T = 4 positions. Parameters: the three projections D by (H d_h) and the output projection (H d_h) by D, with no biases. FLOPs of the score products Q_h K_h^T of all heads, 2 T^2 d_h for each head. Scores held per sequence: H T^2.

| heads H | head width d_h | parameters | FLOPs of all score products | score entries |
|---|---:|---:|---:|---:|
| 1 | 6 | 144 | 192 | 16 |
| 2 | 3 | 144 | 192 | 32 |
| 3 | 2 | 144 | 192 | 48 |
| 6 | 1 | 144 | 192 | 96 |
<!-- /worked -->

The parameters are $3D \cdot (H d_h) + (H d_h)\cdot D = 4D^2$ whatever $H$ is, because the heads share $D$ between them. The FLOPs of the score products are the same too, $H \cdot 2T^2 d_h = 2T^2 D$. What grows with $H$ is the number of scores, $H T^2$, and so the memory of the weights. Heads buy a division of labour, not capacity, and the price is the score memory that the cost section will count.

The table's dimensions are all different on purpose: $D = 6$, $H = 2$, $d_h = 3$ and $T = 4$. If $H$ and $d_h$ were equal, or $T$ and $d_h$, an index of one kind used where the other belongs would give a legal, wrong answer and no check could tell. The same care appears in the next section's gradient check, and in brain's tests.

## The backward pass of attention, one product at a time

A layer that is to be trained needs gradients with respect to the three matrices, the output matrix and its input. The layer is a chain of four operations, each of which Volume III has already differentiated, and the derivation is only the order in which they are undone. Let $d\mathbf{O}$ be the gradient of the loss with respect to the weighted sums (for one head), as it arrives from the layer above. The forward pass saved $\mathbf{Q}$, $\mathbf{K}$, $\mathbf{V}$ and $\mathbf{P}$.

**The average, $\mathbf{O} = \mathbf{P}\mathbf{V}$.** A matrix product, so by its backward rule $d\mathbf{P} = d\mathbf{O}\,\mathbf{V}^\top$ (shape $[T, T]$) and $d\mathbf{V} = \mathbf{P}^\top d\mathbf{O}$ (shape $[T, d_v]$).

**The softmax, $\mathbf{P}_{i:} = \mathrm{softmax}(\mathbf{S}_{i:})$.** Each row on its own. The Jacobian of one row is $\mathrm{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top$, and the gradient row arriving at it, $d\mathbf{p}$, comes out as $d\mathbf{p}\,(\mathrm{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top) = \mathbf{p}\odot d\mathbf{p} - (\mathbf{p}\cdot d\mathbf{p})\,\mathbf{p}$. With $c_i = \sum_j p_{ij}\,dp_{ij}$ the entries of the row term:

$$
dS_{ij} = p_{ij}\,\big(dp_{ij} - c_i\big).
$$

(This is the rule of Chapter 6, now for a whole matrix. It is applied to the scaled scores; the scaling is next.)

**The scores, $\mathbf{S} = \mathbf{Q}\mathbf{K}^\top/\sqrt{d_k}$.** Another matrix product, with $\mathbf{K}^\top$ as its right operand, and a constant factor, so $d\mathbf{Q} = d\mathbf{S}\,\mathbf{K}/\sqrt{d_k}$ and $d\mathbf{K} = d\mathbf{S}^\top\mathbf{Q}/\sqrt{d_k}$: the second is the gradient with respect to $\mathbf{K}^\top$, which is $\mathbf{Q}^\top d\mathbf{S}$, transposed.

**The projections.** $\mathbf{Q} = \mathbf{X}\mathbf{W}_Q$ and its two siblings give $d\mathbf{W}_Q = \mathbf{X}^\top d\mathbf{Q}$, $d\mathbf{W}_K = \mathbf{X}^\top d\mathbf{K}$, $d\mathbf{W}_V = \mathbf{X}^\top d\mathbf{V}$. The input $\mathbf{X}$ was used three times, so its gradient is the sum of three contributions (the rule for a value used in several places):

$$
d\mathbf{X} = d\mathbf{Q}\,\mathbf{W}_Q^\top + d\mathbf{K}\,\mathbf{W}_K^\top + d\mathbf{V}\,\mathbf{W}_V^\top .
$$

For several heads, the output matrix first: $d\mathbf{O}_{\mathrm{cat}} = d\mathbf{Y}\,\mathbf{W}_O^\top$ and $d\mathbf{W}_O = \mathbf{O}_{\mathrm{cat}}^\top d\mathbf{Y}$, and then head $h$ takes the columns $h\,d_h, \dots, (h+1)\,d_h - 1$ of $d\mathbf{O}_{\mathrm{cat}}$ as its own $d\mathbf{O}$. Its $d\mathbf{Q}$, $d\mathbf{K}$, $d\mathbf{V}$ go back into the same column blocks, and the three weight gradients are single products.

```text
function attention_backward(dO: array<real>[T, dv], X: array<real>[T, D],
                            Wq, Wk, Wv, Q, K, V, P)
        -> (array<real>[T, D], array<real>[D, dk], array<real>[D, dk],
            array<real>[D, dv]):
    dP = matmul(dO, transpose(V))           # through the average
    dV = matmul(transpose(P), dO)
    dS = zeros(real, [T, T])
    for i in 0 .. T:
        c = dot(P[i, :], dP[i, :])          # the row term of the softmax rule
        for j in 0 .. T:
            dS[i, j] = P[i, j] * (dP[i, j] - c)
    dQ = matmul(dS, K)
    dK = matmul(transpose(dS), Q)
    for i in 0 .. T:
        for c in 0 .. dk:
            dQ[i, c] = dQ[i, c] / sqrt(dk)  # the factor of the scores
            dK[i, c] = dK[i, c] / sqrt(dk)
    dX = matmul(dQ, transpose(Wq)) + matmul(dK, transpose(Wk))
         + matmul(dV, transpose(Wv))
    return (dX, matmul(transpose(X), dQ), matmul(transpose(X), dK),
            matmul(transpose(X), dV))
```

The mask has no step of its own. A hidden entry has $p_{ij} = 0$, and the softmax rule gives $dS_{ij} = p_{ij}(\dots) = 0$ there, so the hidden pairs send nothing back to $\mathbf{Q}$ or $\mathbf{K}$, and a matrix with the zeros already in the right places is the causal gradient. That is also why a mask is only ever added to the scores in the forward pass.

A derivation of this length has a place for each mistake to hide, and the check of Volume III is the way to find them. Central differences on every parameter of a small model, with the loss of Chapter 3's softmax cross-entropy at the end:

<!-- worked: self_attention.gradient_check -->
Generator: the book's SplitMix64, seed 0x6fd8950bcdf63c2b (`seed_of` of the name `self_attention.check`). T = 5 positions, D = 6, H = 2 heads of width 3, 4 classes, causal mask. Draws, in this order: X (5 by 6, each `normal()`), W_Q, W_K, W_V, W_O (6 by 6) and W_C (6 by 4), each row by row with `0.5 * normal()`, then the 5 target classes `integer(4)`. The model is multi-head attention, then the classifier logits = Y W_C at every position, and the loss is the mean over positions of the softmax cross-entropy of Volume III. Central differences with step 1e-05 on every entry of every parameter and of the input X. Tolerance: each cell is the power of ten that the largest difference stays below (an implementation that adds in another order lands within a factor of ten of it; the criterion is 1e-6); the relative difference |a - n| / max(|a|, |n|) is taken over the entries whose gradient is above 1e-4.

| tensor | entries checked | largest absolute difference below | largest relative difference below |
|---|---:|---:|---:|
| Wq | 36 | 1e-10 | 1e-8 |
| Wk | 36 | 1e-10 | 1e-8 |
| Wv | 36 | 1e-10 | 1e-8 |
| Wo | 36 | 1e-10 | 1e-8 |
| Wc | 24 | 1e-10 | 1e-7 |
| X | 30 | 1e-10 | 1e-8 |
<!-- /worked -->

Every tensor passes, the three projections, the output matrix, the classifier and the input, at a relative difference between $10^{-8}$ and $10^{-7}$, well inside the criterion of $10^{-6}$. The dimensions are $T = 5$, $D = 6$, $H = 2$, $d_h = 3$ and 4 classes, all distinct, so that a swapped axis would not cancel.

A check that passes is worth something only if it can fail. The same check on three wrong backward passes, each a plausible slip in one step:

<!-- worked: self_attention.wrong_backward_passes -->
The check of the previous block (same parameters, same step) run on a correct and on three wrong backward passes. Entries: the largest relative difference |a - n| / max(|a|, |n|) over the entries of the tensor whose gradient is above 1e-4; a difference of 1e-6 or more is written to one decimal, a smaller one as the power of ten it stays below (a tensor that the wrong step does not touch keeps its correct gradient).

| backward pass | W_Q | W_K | W_V | W_O | X |
|---|---:|---:|---:|---:|---:|
| the chapter's backward pass | 1e-8 | 1e-8 | 1e-8 | 1e-8 | 1e-8 |
| softmax rule without the row term | about 1.2 | about 1.3 | 1e-8 | 1e-8 | about 1.6 |
| 1 / sqrt(d_k) left out of dQ and dK | about 0.4 | about 0.4 | 1e-8 | 1e-8 | about 0.7 |
| dO cut into heads as [H, T, d_v] | about 2.0 | about 1.9 | about 1.8 | 1e-8 | about 2.0 |
<!-- /worked -->

Leaving the row term out of the softmax rule gives the right $d\mathbf{V}$ and $d\mathbf{W}_O$, which lie before the softmax in the backward order, and wrong gradients of $\mathbf{W}_Q$, $\mathbf{W}_K$ and the input, which lie after it. Forgetting the scale errs in the same tensors by a smaller factor, about 0.4. Cutting the gradient into heads as if the concatenation were $[H, T, d_v]$ (which is a legal reshape, since the number of elements is the same) corrupts everything that follows the cut, and spares the output matrix, which is applied before it. The pattern of which tensors are wrong says which step is: a hint that costs nothing and is worth reading before reaching for a debugger.

## What the $T \times T$ matrix costs

The score matrix is the price of the all-to-all look, and it is easy to count. For one layer on $T$ positions of width $D$, in the counting rule of Chapter 5 (a product of an $m \times k$ and a $k \times n$ matrix is $2mkn$ FLOPs):

- the three projections and the output projection are four products of $[T, D]$ with $[D, D]$: $4 \cdot 2TD^2 = 8TD^2$;
- the scores $\mathbf{Q}_h\mathbf{K}_h^\top$ of head $h$ are $2T^2 d_h$, and over all heads $2T^2 D$;
- the weighted sums $\mathbf{P}_h\mathbf{V}_h$ are the same again, $2T^2 D$;
- the scaling and the softmax cost 5 FLOPs per score (scale, subtract the largest, exponential, add to the normalizer, divide), $5HT^2$.

The terms of attention itself are $4T^2 D$ plus $5HT^2$, quadratic in $T$, against $8TD^2$, linear.

<!-- worked: self_attention.cost_of_a_layer -->
One attention layer on T positions of width D = 1024 with H = 16 heads of width 64, no biases. Counting rule: a matrix product of an m by k matrix with a k by n matrix is 2 m k n FLOPs; the softmax and the scaling cost 5 FLOPs per score (scale, subtract the largest, exponential, add to the normalizer, divide). The closed forms 8 T D^2 (four projections), 4 T^2 D (the products Q K^T and P V of all heads together) and 5 H T^2 agree with a count made by walking the loops (checked for two small layers, with and without the causal skip of the masked scores).

| T | projection FLOPs 8 T D^2 | score and average FLOPs 4 T^2 D | softmax FLOPs 5 H T^2 | attention products / projections | attention products / whole layer |
|---|---:|---:|---:|---:|---:|
| 512 | 4,294,967,296 | 1,073,741,824 | 20,971,520 | 0.250 | 0.199 |
| 1,024 | 8,589,934,592 | 4,294,967,296 | 83,886,080 | 0.500 | 0.331 |
| 2,048 | 17,179,869,184 | 17,179,869,184 | 335,544,320 | 1.000 | 0.495 |
| 4,096 | 34,359,738,368 | 68,719,476,736 | 1,342,177,280 | 2.000 | 0.658 |
| 8,192 | 68,719,476,736 | 274,877,906,944 | 5,368,709,120 | 4.000 | 0.788 |
| 32,768 | 274,877,906,944 | 4,398,046,511,104 | 85,899,345,920 | 16.000 | 0.924 |

The two products are equal at T = 2 D = 2048 (17,179,869,184 FLOPs against 17,179,869,184). The backward pass of the attention products is four products of the same size (dP, dV, dQ and dK, each 2 T^2 D), so at T = 4096 it costs 137,438,953,472 FLOPs against 68,719,476,736 for the forward products, twice as many. A causal layer that skips the masked scores does T (T + 1) / 2 of the T^2 score computations, 0.5001 of the work at T = 4096.
<!-- /worked -->

![FLOPs of one attention layer of width 1024 against the sequence length T, from 256 to 8192. The four projections grow along a straight line; the two products Q K transposed and P V grow along a parabola that crosses it at T = 2048 and reaches about four times the projections at T = 8192.](img/attention-flops.svg)

Below $T = 2D$ the projections dominate, which is why a model that reads 512 tokens spends most of its arithmetic in matrix products that have nothing to do with attention. Above it the quadratic term takes over: at $T = 32{,}768$ the attention products are 16 times the projections. The softmax is a small part of the FLOPs, 5 per score against $4 d_h$ in the two products, and not a small part of the cost, as the bytes will say. A causal layer that skips the hidden scores computes $T(T+1)/2$ of the $T^2$, half of the products plus a sliver, and the backward pass is four products of the same size as the forward's two (the rule of the last section: $d\mathbf{P}$, $d\mathbf{V}$, $d\mathbf{Q}$, $d\mathbf{K}$), twice the forward products.

The memory is the other count.

<!-- worked: self_attention.score_memory -->
The score matrix of one layer for one sequence with H = 16 heads, in float32 (4 bytes per entry), and the same for 24 layers if the weights of every layer are kept for the backward pass.

| T | entries H T^2 | bytes | GiB, one layer | GiB, 24 layers |
|---|---:|---:|---:|---:|
| 2,048 | 67,108,864 | 268,435,456 | 0.25 | 6.0 |
| 8,192 | 1,073,741,824 | 4,294,967,296 | 4.00 | 96.0 |
| 32,768 | 17,179,869,184 | 68,719,476,736 | 64.00 | 1536.0 |

The chain of three kernels that materialises the matrix writes the scores (4 bytes), reads them to form the softmax, writes the weights (4 bytes) and reads them for the weighted sum: 16 bytes of traffic per score. For a head width of d_h = 64 one score costs 4 d_h + 5 = 261 FLOPs (two products and the softmax), so the chain does 16.31 FLOPs for every byte moved through the matrix.
<!-- /worked -->

At $T = 8192$ the weights of one layer for one sequence take 4 GiB, and 24 layers kept for the backward pass take 96 GiB, before a single parameter, activation of the feed-forward network or optimizer state has been counted. And the matrix is not only held: in the chain of kernels that the last section of this chapter meets, scores are written once, read by the softmax, probabilities written, and read by the average, 16 bytes of traffic per score for $4 d_h + 5$ FLOPs, which is 261 FLOPs for 16 bytes (16.31 FLOPs per byte) at $d_h = 64$. Volume IV's roofline says what such a ratio means on a device whose ridge sits higher: the step is limited by the bytes. Attention is exact, and its cost is quadratic in both time and memory; Chapter 13 shows how to compute the same numbers without ever holding the matrix.

## A layer that learns to look things up

A layer that holds any weights at all should be able to learn something that needs them. The smallest task that requires an attention layer, and cannot be done well without one, is a lookup. A sequence has three key-value pairs and then a query that names one of the keys; the answer is that key's value.

<!-- worked: self_attention.lookup_task -->
Ids 0 .. 15 are the pair tokens `K<k>V<v>` (id 4 k + v: key k in 0 .. 3 paired with value v in 0 .. 3) and ids 16 .. 19 are the query tokens `Q<k>` (id 16 + k), so the vocabulary has V = 20 ids. An example is generated by the draws, in this order: `shuffle([0, 1, 2, 3], rng)`, of which the first three entries are the keys of the three pairs; `integer(4)` three times for their values; `integer(3)` for the index of the key that is asked. The input is the three pair tokens in that order followed by the query token (T = 4); the target is the value of the asked key. Three examples from `self_attention.lookup.examples` (0xfc9769e14cb7b3f5):

| input (T = 4 tokens) | target |
|---|---|
| `K3V2` `K1V1` `K2V1` `Q2` | `V1` |
| `K0V0` `K1V1` `K2V3` `Q0` | `V0` |
| `K1V3` `K3V1` `K2V0` `Q3` | `V1` |

A model that does not use the pairs can only guess one of 4 values: loss ln 4 = 1.3863 nats, accuracy 0.25. A model that sees which values occur among the three pairs but cannot tell which key each belongs to can always answer with the commonest value. The asked pair is one of the three with equal chance, so for a triple of values the best chance is (the count of the commonest value) / 3; averaged over the 64 equally likely triples it is 108 / 192 = 0.5625, the most such a model can reach. The model is the one-layer attention of the text: embedding table E [20, 8] (row by row, `0.5 * normal()`), then W_Q, W_K, W_V [8, 4] and the readout W_O [4, 4] (each row by row, `0.5 * normal()`), 272 parameters in all, causal mask, one head; the logits are the output of the last position times W_O and the loss is the softmax cross-entropy of that position only.
<!-- /worked -->

The task needs exactly what attention gives. The model must find the pair whose key matches the query, which is a comparison of the query token with each pair token, and then carry that pair's value, which is a weighted sum with the weight on one pair. It has no position information, and does not need it: whichever order the pairs come in, the answer is the same, and a layer that treats its input as a set is a natural fit. Without attention, the comparison has nothing to act on, and the experiment replaces the learned weights by uniform ones, the second model, to show how much of the answer depends on the weights.

<!-- worked: self_attention.lookup_training -->
Training: Adam (learning rate 0.01, beta1 0.9, beta2 0.999, epsilon 1e-08, with the bias correction of Volume III), minibatches of 16 fresh examples, 600 updates, no clipping. Examples come from the generator `self_attention.lookup.train.<run>`, so the two models of a run see the same ones; the parameters are drawn from `self_attention.lookup.init.<model>.<run>` (seeds in the table; the uniform model draws the same parameters, including the W_Q and W_K it does not use). The 'uniform' model replaces every attention weight by 1 over the positions the mask allows, so the logits are W_O applied to the mean of the four value vectors x_t W_V of the whole input (weight 1/4 on each of the four positions, the query token included). The parameters that train are E, W_V and W_O; W_Q and W_K stay at their initial values, because the scores do not reach the output and the backward pass sets the gradient of the scores to zero; the gradient of E is the one through W_V alone. A loss column is the mean of the {WINDOW} batch losses that end at that step. Accuracy is the fraction of the 400 test examples of `self_attention.lookup.test` (0xe6260bf46083a7f8) whose largest logit (first on a tie) is the target. Rounding noise grows into any difference in a trained network. A number that comes from a chaotic run is printed with only the digits that are the same in the run and in four copies of it whose first weight is changed by +1e-12, -1e-12, +3e-12 and -3e-12: in full if all agree on it, as 'about x' with the leading digits on which all agree, or as 'unstable' when not even one digit survives; the other digits cannot be reproduced in another implementation.

| model | run | seed of the initial parameters | loss, step 50 | loss, step 150 | loss, step 300 | loss, step 600 | test accuracy |
|---|---|---:|---:|---:|---:|---:|---:|
| attention | 0 | 0x97ca9bcc54861fff | 1.0689 | 0.0910 | 0.0018 | 0.0004 | 1.000 |
| attention | 1 | 0x6a7ad5d56f250cd8 | 1.2636 | 0.2184 | 0.0059 | 0.0005 | 1.000 |
| uniform | 0 | 0x482bdb808d8eae09 | 1.2424 | 0.9686 | 0.8585 | 0.8855 | 0.560 |
| uniform | 1 | 0x1b0c71a58bc985da | 1.2617 | 0.8858 | 0.9124 | 0.9398 | 0.547 |
<!-- /worked -->

The attention models reach a loss of 0.0004 and 0.0005 and classify all 400 test examples correctly in both runs; the loss of run 0 is 1.0689 at step 50, 0.0910 at step 150 and 0.0018 at step 300, so most of the learning happens between steps 50 and 300. The uniform models stop near a loss of 0.9 (against ln 4 = 1.386 for a guess) and reach accuracies of 0.560 and 0.547: well above the 0.25 of a guess, since the mean of the value vectors can encode which values occur among the pairs, and just below the 0.5625 that the task's table gives as the most a model can reach without binding keys to values. Every number in the table survived the stable-digits rule of this book: copies of the run with the first weight changed by amounts of order $10^{-12}$ agree with it on every printed digit, so another implementation that adds in another order should expect the same table.

The weights show what the attention model does:

<!-- worked: self_attention.lookup_attention -->
Run 0 on the example from `self_attention.lookup.shown` (0x137abe287912591e): the input `K1V1` `K2V2` `K3V0` `Q2`, so the target is `V2`. The row of attention weights of the query position (the last) over the positions 1 to 4, and the largest difference of the logits when the three pairs are reordered (pairs 3, 1, 2 in that order, query last). Weights are printed by the stable-digits rule.

| model | weight on pair 1 | pair 2 | pair 3 | the query itself | change of the logits when the pairs are reordered |
|---|---:|---:|---:|---:|---|
| attention | 0.00 | 1.00 | 0.00 | 0.00 | below 1e-9 |
| uniform | 0.25 | 0.25 | 0.25 | 0.25 | below 1e-9 |
<!-- /worked -->

The query `Q2` puts weight 1.00 on `K2V2`, the pair with the same key, and 0.00 on the rest, itself included. That is a lookup: the query vector of `Q2` and the key vector of `K2V2` must have been trained to point the same way, and the value vector of `K2V2` to carry the 2. The uniform model has the weights 0.25 each, as the design required, and cannot tell which value belongs to the key. Both models give the same logits when the three pairs are reordered, to round-off, which is the equivariance of the earlier section made useful: the query is at the last position, so the mask hides nothing from it, and what it sees is a set.

## The attention kernels of brain

Brain's attention is a set of small kernels, one for each step the chapter derived, and they are the chapter's formulas in an order a GPU can run. For a training layer with fused queries, keys and values in one buffer, `crates/kernels/wgsl/attn_scores.wgsl` computes one score per thread: for each batch element, head, query $i$ and key $j$, the dot product of two head-width slices divided by the square root of the head width, and, for $j > i$, a large negative number in place of a score. The kernel writes about the most negative finite 32-bit float, $-3.4 \times 10^{38}$, in place of $-\infty$, so that no infinity ever enters the arithmetic. `attn_softmax.wgsl` handles one row per thread, finds the largest score, sums exponentials over $j \le i$ only and writes exact zeros beyond the diagonal, so that the mask is enforced twice, by the score and by the loop bound. `attn_apply.wgsl` takes the weighted sum of values and writes its result at column $h \cdot \mathrm{head\_dim} + d$ of the output row: the concatenation of heads of this chapter is nothing but that index.

The backward pass is the derivation in the same four steps. `attn_bwd_dscores.wgsl` computes $d\mathbf{P}$ and the softmax rule together, with the row term accumulated in a first loop and the entries in a second (the dot product with the value is simply recomputed instead of being stored), `attn_bwd_dv.wgsl` the sum over $i \ge j$ of $p_{ij}\,d\mathbf{o}_i$, and `attn_bwd_dq.wgsl` and `attn_bwd_dk.wgsl` the two products with the factor $1/\sqrt{d_k}$. The loop bounds are the triangle and its transpose, and there is no kernel for the mask: the zeros of the probabilities are the whole story. All seven kernels, together with a fused one that streams over the keys without a score matrix (`attention.wgsl`), first appear in commit `1fa53524c` (June 2026), whose subject is just "add shaders"; the memory-lean form is the subject of a later chapter, and so are the grouped-query variants, which share each key and value among several queries.

`crates/model/src/block.rs` composes them. `gqa_fwd` dispatches scores, softmax and apply, `gqa_bwd` the four backward steps in the order of the derivation, and the `Gqa` shape structure says in its documentation that ordinary multi-head attention is the case in which the number of key-value heads equals the number of query heads. The bidirectional pair `bidir_fwd` and `bidir_bwd` has no mask and serves encoders. Masks other than the causal one are further score kernels in front of the same softmax: a sliding window in `gqa_scores_win.wgsl`, a prefix mask that lets the first positions see each other in `attn_prefix_mask.wgsl` (commit `5922df133`, July 2026, whose message observes that it needs no backward pass because the softmax already gives a masked entry no gradient), a padding mask in `attn_keypad_mask.wgsl` that removes padded keys for every query, and a score kernel with a caller-supplied constant scale and an additive per-head bias in `attn_scores_bidir_bias.wgsl` (commit `fd1804cde`, July 2026, whose header says that the model it was written for uses a scale of 8 with L2-normalised queries and keys in place of $1/\sqrt{d}$). And `flash_gqa_causal_fwd` is the attention that never holds the matrix, whose documentation states peak memory $O(T \cdot \mathrm{head\_dim})$ in place of $O(T \cdot T)$: that is Chapter 13's subject.

The tests are the chapter's two checks. `crates/model/tests/it/flash_gqa_train.rs` compares the memory-lean forward and backward against the materialised chain above at several head widths (40, 128 and 256), with sequence lengths that are not a multiple of any tile, a batch larger than 1 and grouping ratios of up to 6, and gradient-checks the whole mixer by central differences at pairwise-distinct dimensions, so that "a swapped stride or head index cannot cancel", the lesson of knowledge #4, that degenerate test dimensions hide whole classes of bugs. It also asserts that at least 24 of the 48 sampled gradients are non-trivial, so that the check cannot pass on a backward pass that produces zeros.

Three stories from the history belong to the three places a mask, a scale or a head can go wrong.

**A mask that was not there.** Commit `29e21264f` (August 2026) fixed a time-series model in `crates/kronos` whose trainer used the bidirectional softmax in a layer that the reference model makes causal during training. The message explains why this is the worst kind of bug: inference sees exactly the causal row, so training on non-causal rows teaches every position but the last from context it will never have, "a defect no gradcheck can see, since a gradcheck proves the backward matches the forward, not that the forward is the right function". It was found by comparing against a reference golden with dropout off, and it is the "one step too loose" row of the boundary table, at the scale of a whole model. The fix is one swap of the softmax kernel in `crates/kronos/src/train.rs`, and the comment beside it states the point of the backward section: the scores kernel can stay bidirectional because the zeros in the probabilities make the bidirectional backward exactly right. The regression test `dependency_layer_attention_is_causal_in_training` in `crates/kronos/tests/train_parity.rs` asserts that the largest weight on a future key is exactly 0. A neighbouring commit, `0893002c3` (August 2026), found the opposite fault in what is now `crates/mimi`, an audio codec: a sliding-window mask whose configuration was parsed and never dispatched. It had stayed invisible because every test used sequences shorter than the window, the same lesson as the degenerate dimensions.

**A head that was dropped from the tape.** Commit `e10959f00` (August 2026) fixed a gradient in `crates/vae`: the tape of attention recorded its channels and length but not the head split, and the backward pass in `crates/vae/src/blocks/grad.rs` rebuilt it with one head. The message says why nothing caught it: the backward kernels are parameterised by head count and head width exactly like the forward ones, so the buffer shapes still line up and every dispatch succeeds, "only the numbers are wrong". It is the third row of the wrong-backward table, found in the code and not by the table, and its remedy is the one the table shows, a gradient check at a configuration with more than one head.

**A scale placed once.** In commit `72a840fdf` (September 2026), a port of a forecasting model whose reference scales its attention in two separate places, by factors that cancel exactly, folds the whole scale into the query's normalization gain when the weights are loaded and calls the scores kernel with a scale of 1.0 everywhere, so that, in the words of the module documentation of `crates/timesfm3/src/model.rs`, "no separate scale parameter exists anywhere downstream to get wrong". The documentation gives the reason: a dropped or doubled scale cannot be seen by cosine similarity, the lesson recorded in knowledge #2. The saturation table of this chapter is why that matters: a missing $1/\sqrt{d_k}$ changes how much weight lands on the winning key, and a comparison that measures only the direction of the outputs misses it.

All of the three went wrong in the places the chapter marks, where the mask, the scale and the head split are applied; none of them was in the softmax itself, which in every kernel subtracts the largest score before exponentiating, as the stable softmax of Volume III does.

## Check your understanding

1. A student builds a causal language model from this chapter's attention and trains it on text. The training loss is far below that of any n-gram model on the same text and the model produces gibberish when it generates. Which single line of the attention is the first suspect, and what test would confirm it in one forward pass?
2. Queries and keys have width $d_k = 64$ with unit-variance entries. What is the standard deviation of an unscaled score, and by what factor is the weight ratio of two keys whose scores differ by one standard deviation larger than it would be after the scale?
3. A colleague says: "multi-head attention with 8 heads has 8 times the parameters of one head." Find the error, and give the parameter count of a layer with $D = 512$.
4. Attention is equivariant to a permutation of its positions. Why does a causal mask break this, and why is that not a good way to give a model its sense of order?
5. The backward pass of a layer gives the correct gradient for $\mathbf{W}_V$ and $\mathbf{W}_O$ and a wrong one for $\mathbf{W}_Q$, $\mathbf{W}_K$ and the input. Which step of the derivation is the suspect, and why can the other two tensors still be right?
6. A model with a sequence length of 32,768 and 16 heads runs out of memory in the attention layer although its parameters fit. Which quantity of the cost section explains it, and what would the memory be for one sequence and one layer in float32?
7. In the lookup experiment the query token is last and the mask is causal. If the query came first and the mask stayed causal, what accuracy could any model reach, and what does that say about the mask?

## Summary

Self-attention lets every position of one sequence read every other in one step. Each position's input vector gives three vectors by three learned maps, a query, a key and a value; the dot products of all queries with all keys, divided by $\sqrt{d_k}$, are the scores; a softmax on each row turns them into weights; and the weighted sums of the values are the outputs. The division by $\sqrt{d_k}$ keeps the scores at unit variance, which was derived and then measured: without it a row of 8 keys puts more than 0.99 of its weight on one key in 290 of 500 rows at $d_k = 256$, and the gradient through the softmax almost vanishes. The layer has no notion of order, as the shuffle test showed, and the causal mask, minus infinity above the diagonal, gives it the one-way flow that a language model needs and lets all next-token problems of a sequence be solved in one pass, with five sequential stages where a recurrent layer has $T$. Several heads run the same layer on slices of the width, at the same parameters and score FLOPs and $H$ times the score memory. The backward pass is the four products and a softmax rule in reverse, it needs no step for the mask, and it passed a central-difference check and three deliberate wrong versions failed it in the tensors the mistake should affect. The cost is $8TD^2$ FLOPs in the projections and $4T^2 D$ in the score and average products, equal at $T = 2D$, with a score matrix of $H T^2$ entries. A one-layer model learned the lookup of a value by its key, which the same model with uniform weights could not. The next chapter confronts the other half of what the shuffle test said: this layer does not know where anything is, and something must tell it.

## References

- A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser and I. Polosukhin, "Attention is all you need", 2017.
- D. Bahdanau, K. Cho and Y. Bengio, "Neural machine translation by jointly learning to align and translate", 2015.

## Exercises

1. **[Calculate]** Three positions have queries $\mathbf{q}_1 = (1, 0)$, $\mathbf{q}_2 = (0, 1)$, $\mathbf{q}_3 = (1, 1)$, keys $\mathbf{k}_1 = (2, 0)$, $\mathbf{k}_2 = (0, 2)$, $\mathbf{k}_3 = (2, 2)$ and values $\mathbf{v}_1 = (1, 0)$, $\mathbf{v}_2 = (0, 1)$, $\mathbf{v}_3 = (1, 1)$ ($d_k = 2$). Give the weight matrix $\mathbf{P}$ without a mask to three decimals and the output of position 3, then with the causal mask.
2. **[Calculate]** For $d_k = 64$, give the standard deviation of an unscaled score, and the ratio of the weights of two keys whose unscaled scores differ by one standard deviation, with and without the division by $\sqrt{d_k}$.
3. **[Calculate]** One row has weights $\mathbf{p} = (0.6, 0.3, 0.1)$ and receives $d\mathbf{p} = (0.5, -1, 2)$. Compute $d\mathbf{s}$ by the softmax rule and check that its entries add to 0. Why must they?
4. **[Calculate]** A layer has $D = 768$, $H = 12$ heads and $T = 1024$. Give its projection FLOPs, its score and average FLOPs, the number of scores and the bytes of the weights of one layer in float32.
5. **[Calculate]** How many scores does a causal layer need at $T = 6$, and what fraction of a full layer's scores is that? Give the limit of the fraction as $T$ grows.
6. **[Calculate]** A layer has $D = 512$ and $H = 8$. Give the number of parameters of the four projections without biases, and the number in one head's three matrices.
7. **[Implement]** Implement attention as the pseudocode of the chapter and reproduce `self_attention.weights_and_outputs`: the generator, the draw order, the weights of row 2 and its output to three decimals, then the masked weights of `self_attention.causal_weights`.
8. **[Implement]** Implement multi-head attention with its backward pass and reproduce `self_attention.gradient_check` (relative differences of order $10^{-8}$ on every tensor), then add the three wrong backward passes and reproduce `self_attention.wrong_backward_passes`, in which the wrong tensors differ from the right ones by amounts that agree to one decimal.
9. **[Implement]** Train the one-layer attention model of the lookup task by the protocol of `self_attention.lookup_training` (generators, draw order, Adam settings), and reproduce its loss at steps 50, 150, 300 and 600 and its test accuracy for run 0, then the model with uniform weights.
10. **[Implement]** Reproduce `self_attention.mask_boundary`: the positions that change when token 3 is replaced under the four masks.
11. **[Investigate]** Train the lookup model without the factor $1/\sqrt{d_k}$ in the scores at $d_k = 4, 16, 64$ (change the width of $\mathbf{W}_Q$ and $\mathbf{W}_K$), and compare the loss after 50 steps with that of the scaled model. At which width does the missing scale first cost anything, and does it ever prevent the model from learning?
12. **[Investigate]** Make the lookup task need two lookups, a query that asks for the value of one key and the value of another, answered at two final positions, and train a model with one head and one with two. What does the second head do, and does one head with a larger $d_k$ do as well?

## Key terms

- **query, key and value**: the three vectors that each position's input gives by three separate learned linear maps: the query is what the position asks of the others, the key is what it can be matched on, and the value is what it hands over when matched. <!-- concept: query-key-value -->
- **scaled dot-product attention**: the layer in which every query is compared with every key by a dot product divided by the square root of their width, a softmax along each row turns the scores into weights, and the weighted sums of the values are the output, $\mathrm{softmax}(\mathbf{Q}\mathbf{K}^\top/\sqrt{d_k})\mathbf{V}$. <!-- concept: scaled-dot-product-attention -->
- **causal mask**: the matrix of zeros and minus infinities added to the scores before the softmax so that position $i$ gives no weight to positions after it, which makes the output at each position a function of its prefix alone and lets all positions of a sequence be trained in one pass. <!-- concept: causal-mask -->
- **multi-head attention**: attention run $H$ times on slices of the width, each head with its own queries, keys and values, the outputs placed side by side and mixed by one output matrix, so that a position can ask several different questions of the same set at once. <!-- concept: multi-head-attention -->
- **attention backward**: the gradient of attention, obtained by undoing in order the weighted average, the softmax rule on each row, the score product with its scale and the three projections, with the input collecting three contributions and no step of its own for the mask. <!-- concept: attention-backward -->

## Answers

**Recall questions.**

1. The decoder's state, turned into a vector by $\mathbf{W}_a$, was the question; the encoder state transformed by $\mathbf{U}_a$ was the key it was compared with to give the score; and the encoder state itself was averaged with the weights.
2. $\exp(0, 0, \ln 3) = (1, 1, 3)$, sum 5, so the weights are $(0.2, 0.2, 0.6)$. The loss of the third class is $-\ln 0.6 = 0.511$ and the gradient with respect to the scores is the probabilities minus the one-hot label, $(0.2, 0.2, -0.4)$.
3. The product has shape $[4, 2]$ and takes $4 \cdot 2 \cdot 3 = 24$ multiplications.
4. $d\mathbf{A} = d\mathbf{C}\,\mathbf{B}^\top$ and $d\mathbf{B} = \mathbf{A}^\top d\mathbf{C}$.
5. The variances add: variance 16, standard deviation 4.
6. 100 steps must run one after another, since each needs the state of the one before. More processors do not shorten that: they do more of each step at once, not more steps.

**Check your understanding.**

1. The mask: the line that adds $-\infty$ above the diagonal. A model that sees the token it predicts has a loss far below what the text allows and cannot generate, which has no future to read. The test is the perturbation test of Chapter 4: replace one input token in one forward pass and list the outputs that change; any change at a position before the replaced token is a leak.
2. The standard deviation is $\sqrt{64} = 8$. The weight ratio for a gap of one standard deviation is $e^8 \approx 2981$ without the scale and $e^1 = 2.718$ with it, so the unscaled ratio is larger by $e^7 \approx 1097$.
3. The heads share the width: each has matrices of shape $[D, D/H]$, so the three projections of all heads together hold $3D^2$ numbers whatever $H$ is, and with the output matrix the layer has $4D^2$. For $D = 512$ that is $4 \cdot 262{,}144 = 1{,}048{,}576$ parameters, for 1 head or for 8.
4. With a mask, position $i$ sees only $1, \dots, i$, so shuffling the inputs changes which positions each output can see and the outputs are no longer a relabelling of one another (the example's difference was 1.608). Some order information is therefore present, but only as how many positions are visible, which cannot tell the order of those that are; it does not replace telling the model where each token is.
5. The step after the softmax rule in the backward order, in particular the row term of the softmax rule: $d\mathbf{V}$ comes from $\mathbf{P}^\top d\mathbf{O}$ and $d\mathbf{W}_O$ from the output of the heads, and neither uses $d\mathbf{S}$, while $d\mathbf{Q}$, $d\mathbf{K}$ and through them $d\mathbf{W}_Q$, $d\mathbf{W}_K$ and the input all do.
6. The weights of the scores, $H T^2$ entries: $16 \cdot 32{,}768^2 = 17{,}179{,}869{,}184$ entries, 68,719,476,736 bytes, 64 GiB, for one sequence and one layer. Memory grows as $T^2$ while the parameters do not depend on $T$.
7. At most 0.25, a guess. Under the causal mask position 1 sees only itself, so a query in first place cannot see any pair, and the answer is read at the position of the query. The mask decides what each position may use, and the experiment places the query where it can use everything.

**Exercises.**

1. Scores $\mathbf{Q}\mathbf{K}^\top$: row 1 $(2, 0, 2)$, row 2 $(0, 2, 2)$, row 3 $(2, 2, 4)$; divided by $\sqrt 2 = 1.4142$: row 1 $(1.414, 0, 1.414)$, row 2 $(0, 1.414, 1.414)$, row 3 $(1.414, 1.414, 2.828)$. Softmax: row 1: $\exp = (4.113, 1, 4.113)$, sum 9.226, weights $(0.446, 0.108, 0.446)$; row 2 $(0.108, 0.446, 0.446)$; row 3: $\exp = (4.113, 4.113, 16.919)$, sum 25.145, weights $(0.164, 0.164, 0.673)$. Output of position 3: $0.164\,(1, 0) + 0.164\,(0, 1) + 0.673\,(1, 1) = (0.836, 0.836)$. With the causal mask, row 1 is $(1, 0, 0)$, row 2 is the softmax of $(0, 1.414)$: $(0.196, 0.804, 0)$, row 3 is as before, and the output of position 3 is unchanged, $(0.836, 0.836)$.
2. The unscaled standard deviation is $\sqrt{64} = 8$; the weight ratio of two keys one standard deviation apart is $e^8 \approx 2981$ unscaled and $e^1 = 2.718$ scaled.
3. $c = 0.6 \cdot 0.5 + 0.3 \cdot (-1) + 0.1 \cdot 2 = 0.2$, so $d\mathbf{s} = (0.6 \cdot 0.3, 0.3 \cdot (-1.2), 0.1 \cdot 1.8) = (0.18, -0.36, 0.18)$, which adds to 0. It must: adding the same number to every score does not change a softmax, so the loss does not change along $(1, 1, 1)$ and the gradient is orthogonal to it.
4. Projections $8 \cdot 1024 \cdot 768^2 = 4{,}831{,}838{,}208$ FLOPs; score and average $4 \cdot 1024^2 \cdot 768 = 3{,}221{,}225{,}472$ FLOPs; scores $12 \cdot 1024^2 = 12{,}582{,}912$; bytes $4 \cdot 12{,}582{,}912 = 50{,}331{,}648$ (48 MiB).
5. $6 \cdot 7 / 2 = 21$ scores of 36, a fraction of $7/12 = 0.583$. In general $(T + 1)/(2T)$, which tends to $1/2$.
6. $4 \cdot 512^2 = 1{,}048{,}576$ for the four projections; one head has $d_h = 64$ and three matrices of $512 \cdot 64 = 32{,}768$ each, 98,304 numbers.
7. The weights of row 2 are $(0.326, 0.047, 0.032, 0.595)$ and its output $(-0.044, -0.252)$; the masked weights of row 2 are $(0.874, 0.126, 0, 0)$.
8. Relative differences of order $10^{-8}$ on every tensor, for the correct pass; the wrong passes differ by about 1.2, 0.4 and 2.0 as the table shows, in the tensors named there.
9. The attention model's loss falls from about 1.07 at step 50 to about 0.0004 at step 600 and its accuracy is 1.000; the uniform model's loss stops near 0.9 and its accuracy is 0.560 for run 0. Another implementation that adds in another order should reproduce the printed digits.
10. No mask: 1, 2, 3, 4; causal: 3, 4; one too loose: 2, 3, 4; one too tight: position 1 is undefined.
11. An investigation. Expect the unscaled model to start slower as $d_k$ grows, as the saturation table predicts, and to learn nonetheless at these small widths; the chapter did not run it.
12. An investigation. Expect the second head to take the second key; whether one wider head can do it depends on whether one softmax row can put weight on two positions at once, which it can, so the difference may be smaller than the constructed example suggests. The chapter did not run it.
