<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Attention Without the Quadratic Memory

*How can attention over long sequences be computed exactly without ever storing the full attention matrix?*

[Chapter 7](07-self-attention.md) ended its cost section on a number worth reading twice. One attention layer with 16 heads on a sequence of 32,768 tokens holds a score matrix of 64 GiB in float32, for one sequence and one layer. The numbers that layer is computed from, the queries, keys and values, and the output it produces are four matrices of 32,768 rows and 1,024 columns, 512 MiB between them. The matrix is 128 times larger than everything that goes in and out of the layer, and it exists, in the chain of three kernels that Chapter 7 and Volume IV's chapter on fusion described, only because the kernels are separate programs and memory is the one way to hand a result from one to the next. Nobody asked for it.

[Chapter 8 of Volume IV](../v04-computation/08-fusion.md) counted what a single fused kernel would save and left the algorithm open: keep a running maximum and a running sum for each row, correct them as blocks of keys arrive, and no workgroup ever needs the whole row of scores at once. This chapter makes that sentence exact. It derives the update, shows that it gives the same attention as the materialised version to rounding error, counts the bytes it moves and the arithmetic it adds, runs it as a kernel on the simulator of Volume IV, extends it to the causal mask and to the backward pass, and then turns it on the opposite shape of the problem, one query against a long cache, where the parallelism has to be found somewhere else. Nothing here is an approximation. The claim is that the matrix is a way of organizing the computation and not a part of its result.

Six ideas from earlier chapters carry the argument. Try them from memory; the answers are at the end of the chapter.

1. (Volume V, [Chapter 7](07-self-attention.md), attention backward.) In the softmax rule $dS_{ij} = p_{ij}(dp_{ij} - c_i)$ of attention, what is the row term $c_i$, and which of $d\mathbf{Q}$, $d\mathbf{K}$ and $d\mathbf{V}$ does it affect?
2. (Volume III, [Chapter 3](../v03-neural-networks/03-softmax.md), stable softmax.) The softmax of $(1000, 1001)$ cannot be formed by exponentiating the inputs. What is done instead, and what does it cost in passes over the vector?
3. (Volume I, [Chapter 15](../v01-mathematics/15-floating-point.md), log-sum-exp.) Compute $\mathrm{lse}(0, \ln 3)$, and say how the sum of exponentials of a long vector is kept from overflowing.
4. (Volume IV, [Chapter 6](../v04-computation/06-matmul.md), tiled matmul.) Two $n \times n$ matrices are multiplied by workgroups that own $T \times T$ output tiles. How many numbers are loaded from global memory, and why is that fewer than $2n^3$?
5. (Volume IV, [Chapter 4](../v04-computation/04-gpu.md), shared memory, with the barrier rule of [Chapter 5](../v04-computation/05-compute-kernels.md).) A workgroup stages a tile in shared memory, uses it, and stages the next. Why does the loop need two barriers?
6. (Volume IV, [Chapter 7](../v04-computation/07-reductions.md), parallel reduction.) In float32, what are $(1 + 2^{-24}) + 2^{-24}$ and $1 + (2^{-24} + 2^{-24})$, and what does that say about summing the same numbers in a tree and in a loop?

## Why the row has to be finished first

Put the matrix next to what it is computed from:

<!-- worked: flash_attention.matrix_versus_inputs -->
One attention layer with H = 16 heads of width d_h = 64 (D = 1024) in float32 (4 bytes per number). The score matrix of all heads has H T^2 entries; the inputs and result are four matrices of shape [T, D] (Q, K, V and the output O). Chapter 7 gave the same matrix for 24 layers; here one layer is set against the numbers it is computed from.

| T | score matrix, bytes | GiB | Q, K, V and O, bytes | MiB | matrix / (Q, K, V, O) |
|---|---:|---:|---:|---:|---:|
| 2,048 | 268,435,456 | 0.25 | 33,554,432 | 32 | 8.0 |
| 8,192 | 4,294,967,296 | 4.00 | 134,217,728 | 128 | 32.0 |
| 32,768 | 68,719,476,736 | 64.00 | 536,870,912 | 512 | 128.0 |
<!-- /worked -->

The last column is $T/(4 d_h)$. At $T = 2048$ the matrix is eight times its inputs, at $T = 32{,}768$ a hundred and twenty-eight times, and it keeps growing with the square of the length while the inputs grow with the length. The matrix is also temporary in the plainest sense. In the forward pass, three kernels write it twice (as scores, then as weights) and read it twice, and when the output is done nothing in the forward pass needs it. (The backward pass wants the weights again; the section on it returns to that.)

Why does it exist? Look at one row $i$, with the scores $s_{ij} = \mathbf{q}_i \cdot \mathbf{k}_j / \sqrt{d_h}$ for the keys $j = 0, \dots, T-1$ (positions are numbered from 0 here, as in the pseudocode). The output of the row is

$$
\begin{aligned}
\mathbf{o}_i &= \sum_{j} p_{ij}\,\mathbf{v}_j, & p_{ij} &= \frac{e^{s_{ij} - m_i}}{\ell_i}, \\
m_i &= \max_j s_{ij}, & \ell_i &= \sum_j e^{s_{ij} - m_i}.
\end{aligned}
$$

The subtraction of $m_i$ is the stable softmax of Volume III, and it is what forces the order of the work. The weight of the very first key cannot be written down until the largest score of the whole row is known, and the normalizer $\ell_i$ cannot be formed until the maximum is known, so a direct implementation makes three passes over the row: the maximum, then the sum, then the weighted sum of values. Each pass needs the whole row at hand, and a row of $T$ scores for each of $T$ queries is the matrix.

The way out is to ask whether the passes can be fused into one that sees each score once as it is produced and keeps only a few numbers about the row. Two of the three passes are sums, and sums can be accumulated while the terms stream by. What cannot be accumulated as it stands is a sum of exponentials taken relative to a maximum that is still changing.

## A softmax that reads its row in pieces

Suppose the first $n$ scores have been read, and keep two numbers about them: their maximum $m$ and the sum $\ell = \sum_{j<n} e^{s_j - m}$ of their exponentials relative to that maximum. A new chunk of scores arrives. Let $m'$ be the larger of $m$ and the largest score of the chunk. The sum we want afterwards is taken relative to $m'$:

$$
\ell' = \sum_{j<n} e^{s_j - m'} + \sum_{c \in \text{chunk}} e^{s_c - m'}.
$$

The second sum is computable from the chunk alone. The first is the old sum with a different reference point, and moving the reference point is a multiplication, because $e^{s_j - m'} = e^{s_j - m}\,e^{m - m'}$ for every $j$:

$$
\ell' = \ell\,e^{m - m'} + \sum_{c \in \text{chunk}} e^{s_c - m'}.
$$

That is the whole idea, and it is exact. The factor $e^{m - m'}$ is at most 1, because $m' \ge m$, so the correction can only shrink the old sum and never overflows; it equals 1 when the chunk did not raise the maximum and it is the correction that re-references the sum when it did. The **online softmax** is this update: carry the pair $(m, \ell)$ for each row, start from $m = -\infty$ and $\ell = 0$ (with $e^{-\infty} = 0$, so that the first chunk multiplies the empty sum by zero), apply the update to each chunk as it arrives, and at the end the weights are $p_j = e^{s_j - m}/\ell$ with the final pair. A chunk of one score is the algorithm of Milakov and Gimelshein (2018); larger chunks cost one rescale per chunk instead of one per score.

Take one row of eight scores in chunks of two:

<!-- worked: flash_attention.online_row -->
The row of eight scores s = (2.0, -1.5, 0.5, 3.0, 1.0, 4.5, -0.5, 4.0), read in chunks of 2. The state starts at m = -infinity and l = 0; per chunk, m' = max(m, largest score of the chunk), the correction is exp(m - m'), and l' = l exp(m - m') + the sum of exp(s - m') over the chunk. exp is the library exp of double precision; the last column is printed to six decimals.

| chunk | scores | m | exp(m_old - m) | l |
|---|---:|---:|---:|---:|
| 0 | (2.0, -1.5) | 2.0 | 0.000000 | 1.030197 |
| 1 | (0.5, 3.0) | 3.0 | 0.367879 | 1.461073 |
| 2 | (1.0, 4.5) | 4.5 | 0.223130 | 1.356207 |
| 3 | (-0.5, 4.0) | 4.5 | 1.000000 | 1.969476 |

Weights from the final state, exp(s_j - m) / l, are (0.041679, 0.001259, 0.009300, 0.113294, 0.015333, 0.507749, 0.003421, 0.307966). The ordinary stable softmax of the whole row gives the same weights; the largest difference over the eight is below 1e-12.
<!-- /worked -->

The maximum grows in the first three chunks and the correction is $e^{-1}$ and $e^{-1.5}$ when it does; in the last chunk $-0.5$ and $4.0$ are both below the running maximum $4.5$, the factor is exactly 1, and the update is a plain addition. The weights computed from the final state are the weights of the stable softmax of the whole row to rounding error, as printed: the sum in a different order and with different references has the same value, and a difference of $10^{-16}$ is the rounding of double precision.

## Carrying the weighted sum along

The output needs more than the weights, it needs $\sum_j p_{ij}\mathbf{v}_j$. The same move works for the numerator. Keep $\mathbf{o} = \sum_{j<n} e^{s_j - m}\mathbf{v}_j$, the sum of the values weighted by the unnormalized exponentials, relative to the running maximum, and on a new chunk

$$
\mathbf{o}' = \mathbf{o}\,e^{m - m'} + \sum_{c \in \text{chunk}} e^{s_c - m'}\,\mathbf{v}_c .
$$

The reasoning is the same line: moving the reference point multiplies the old sum by $e^{m - m'}$. The row's output is $\mathbf{o}/\ell$ at the end. So the whole state of a row is the triple $(m, \ell, \mathbf{o})$, one number, one number and one vector of the width of a value, and a row never needs more than that however many keys it has read:

<!-- worked: flash_attention.online_values -->
The same row with a value of width 2 for each score, v_0 .. v_7 = (1.0, 0.0), (0.0, 1.0), (2.0, 1.0), (-1.0, 2.0), (0.5, -1.0), (1.0, 1.0), (-2.0, 0.0), (0.0, -2.0). The state is (m, l, o) with o the running weighted sum, rescaled by exp(m - m') whenever the maximum grows. The last two columns compare o / l with the attention output over the prefix read so far, softmax of the first scores applied to the first values; the bound is the power of ten the largest difference stays below.

| chunk | m | l | o | o / l | differs from prefix attention by less than |
|---|---:|---:|---:|---:|---:|
| 0 | 2.0 | 1.030197 | (1.000000, 0.030197) | (0.970688, 0.029312) | 1e-12 |
| 1 | 3.0 | 1.461073 | (-0.467951, 2.093194) | (-0.320279, 1.432641) | 1e-12 |
| 2 | 4.5 | 1.356207 | (0.910685, 1.436857) | (0.671494, 1.059468) | 1e-12 |
| 3 | 4.5 | 1.969476 | (0.897209, 0.223796) | (0.455557, 0.113632) | 1e-12 |

After the last chunk, o / l = (0.455557, 0.113632); the materialised softmax(s) V is (0.455557, 0.113632), and the two agree to below 1e-12.
<!-- /worked -->

The last columns show a property that is worth more than it looks. After each chunk, $\mathbf{o}/\ell$ is exactly the attention output of the keys read so far, as if the sequence had ended there. The state is not an approximation that becomes exact at the end; it is exact for the prefix at every step.

The triple has a second property that the rest of the chapter uses. Let the state of a set $A$ of keys be $(m_A, \ell_A, \mathbf{o}_A)$, defined as above from that set alone. For two disjoint sets the state of their union follows by the identity that moved the reference point:

$$
\begin{aligned}
m &= \max(m_A, m_B), \\
\ell &= \ell_A e^{m_A - m} + \ell_B e^{m_B - m}, \\
\mathbf{o} &= \mathbf{o}_A e^{m_A - m} + \mathbf{o}_B e^{m_B - m}.
\end{aligned}
$$

Call this the **merge** of two states. The update of the online softmax is the merge of the running state with the state of the chunk. The merge gives the same result in any order and any grouping, because each way of applying it ends at the state of the union of all the sets, and it has an identity, the empty state $(-\infty, 0, \mathbf{0})$. It is therefore an associative operator on a small record, the situation of Volume IV's reductions with the record in the place of the number: a tree of merges is as legitimate as a loop of them, and the pieces can be summed by different workgroups and combined afterwards.

<!-- worked: flash_attention.merge_demo -->
The row of the previous blocks cut into three sets of keys, each summarised by its own state (m, l, o) from the empty state, then merged: m = max(m_A, m_B), l = l_A exp(m_A - m) + l_B exp(m_B - m), and o likewise.

| states | m | l | o |
|---|---:|---:|---:|
| keys 0 .. 2 | 2.0 | 1.253328 | (1.446260, 0.253328) |
| keys 3 .. 4 | 3.0 | 1.135335 | (-0.932332, 1.864665) |
| keys 5 .. 7 | 4.5 | 1.613269 | (0.986524, -0.213061) |
| (A + B) + C | 4.5 | 1.969476 | (0.897209, 0.223796) |
| A + (B + C) | 4.5 | 1.969476 | (0.897209, 0.223796) |
| (C + A) + B | 4.5 | 1.969476 | (0.897209, 0.223796) |
| all eight in one pass | 4.5 | 1.969476 | (0.897209, 0.223796) |

The three orders of merging give the state of the single pass; the largest difference of the outputs o is below 1e-12 and that of l below 1e-12.
<!-- /worked -->

Any grouping gives the state of the single pass, which is why the keys of one row may be split among workgroups. (The groupings are not the same sequence of floating-point operations, so the results agree to rounding error and not to the bit, a point that the section on numerics returns to.)

## Tiling the loop and counting what it moves

The online softmax says that a row may be read in pieces. A GPU makes that useful by reading pieces for many rows at once. **Tiled attention** is the schedule: a workgroup takes a block of $B_q$ query rows, keeps their running states and their queries on chip, and walks along the keys in tiles of $B_k$ rows; for each tile it loads the keys and values into shared memory, forms the $B_q \times B_k$ block of scores, folds the block into the states of its rows, and moves on. The scores of a tile exist for the length of one step and are never written to global memory. This is the tiling of Volume IV, Chapter 6, applied not to one matrix product but to the whole chain of two products with a softmax between them, which the online softmax makes possible.

![The loop of tiled attention for one workgroup: the query block is read once from global memory into registers, the key and value tiles stream through shared memory one after another, the tile of scores only ever exists on chip and updates the running state m, l, o, and the output block and the log-sum-exp are written once at the end.](img/flash-tile-loop.svg)

Count the global-memory traffic of one head, in numbers moved ($T$ rows of width $d_h$ for each matrix, $B_q$ query rows per block). The query block is read once, so $Q$ costs $T d_h$ in all. The keys and values are read again for every query block, $\lceil T/B_q \rceil$ times, $2T d_h$ each time. The output costs $T d_h$ and the saved log-sum-exp (it comes back with the backward pass) costs $T$:

$$
\begin{aligned}
\text{words tiled} &= 2T d_h + 2T d_h\,\Big\lceil \frac{T}{B_q} \Big\rceil + T \\
&\approx \frac{2T^2 d_h}{B_q}.
\end{aligned}
$$

The three-kernel chain moves $4T^2 + 4T d_h$ words (the count of Volume IV, Chapter 8: write $S$, read it, write $P$, read it, plus the inputs and the output). The two counts as bytes, for one head of width 64:

<!-- worked: flash_attention.bytes_moved -->
Bytes moved to and from global memory by one head of width d_h = 64 in float32, in MiB (1 MiB = 1,048,576 bytes), as a function of the sequence length T. Materialised: 16 T^2 + 16 T d_h bytes (the count of Volume IV, Chapter 8; asserted equal to the unfused count of `fusion.attention_bytes`). Chain with tiled products: the same three kernels when the two matrix products are tiled kernels of output tile 64 (loads k (m ceil(n / 64) + n ceil(m / 64)) for each, Volume IV, Chapter 6). Tiled attention: 4 (2 T d_h + 2 T d_h ceil(T / B_q) + T) bytes, the query block read once, K and V read again for every query block, the output and the log-sum-exp written once.

| T | materialised | chain, tiled products | tiled, B_q = 16 | tiled, B_q = 64 | tiled, B_q = 256 |
|---|---:|---:|---:|---:|---:|
| 1,024 | 17.0 | 28.2 | 32.5 | 8.5 | 2.5 |
| 4,096 | 260.0 | 449.0 | 514.0 | 130.0 | 34.0 |
| 16,384 | 4,112.0 | 7,172.0 | 8,200.1 | 2,056.1 | 520.1 |
| 65,536 | 65,600.0 | 114,704.0 | 131,104.2 | 32,800.2 | 8,224.2 |

Tiled words over materialised words at T = 16384: B_q = 16: 1.994; B_q = 32: 0.998; B_q = 64: 0.500; B_q = 256: 0.126.
<!-- /worked -->

![Gibibytes moved to and from global memory by one head of width 64 as the sequence length grows from 1,024 to 16,384: the chain that materialises S and P, and tiled attention with 16, 64 and 256 query rows per block. Every curve is quadratic; the tiled curves are the chain's divided by a constant that grows with the block: twice as high for 16 rows, half for 64, an eighth for 256.](img/flash-bytes-vs-length.svg)

Two lessons sit in the table. The first is that the saving is not automatic. Both counts are quadratic in $T$; what differs is the constant, and the tiled one is $2 d_h/B_q$ against 4. With $d_h = 64$ the two are equal at $B_q = 32 = d_h/2$, and with 16 query rows per block tiled attention moves twice the bytes of the chain. (A chain whose two matrix products are themselves tiled kernels, the third column, reads their operands again and again and moves about $7T^2$ words and not $4T^2$; the comparison with $4T^2$ is the one that favours the chain.) The second is what the saving is made of: *more query rows per block*. Every doubling of $B_q$ halves the traffic of the keys and values, since each tile that was loaded is used by twice as many rows.

How big can $B_q$ be? It is bounded by the fast memory that holds the block. Let $M$ be the number of floats of fast memory a workgroup has for the following: its $B_q$ query rows, the $B_q$ rows of output it is accumulating, a key tile and a value tile of $B_k$ rows, and the $B_q \times B_k$ scores. Take $B_q = B_k = B$. The footprint is $4B d_h + B^2 \le M$, and for $B \ll d_h$ the first term dominates, so $B \approx M/(4d_h)$ and

$$
\text{words tiled} \approx \frac{2T^2 d_h}{B} = \frac{8\,T^2 d_h^2}{M}.
$$

The traffic is $\Theta(T^2 d_h^2 / M)$: quadratic in $T$ and in $d_h$, divided by the size of the fast memory. Dao and coauthors (2022), who named the method FlashAttention, derive this count and state that no exact algorithm can do asymptotically better over a wide range of $M$; this chapter takes the lower bound on trust and derives only the upper one. Rabe and Staats (2021) had shown that the memory of the attention computation need not grow with the square of the length; what the 2022 paper added was the accounting of reads and writes of the slow memory, and a kernel built on it. Here are the tiles that fit two sizes of memory:

<!-- worked: flash_attention.tile_choice -->
The tile that fits in M floats of fast memory: the largest B with 4 B d_h + B^2 <= M (a query block, an output block, a key tile and a value tile of B rows each, and the B by B scores), for T = 16384 and the two memory sizes of 48 and 96 KiB (M = 12,288 and 24,576 floats). 'Words tiled' is 2 T d_h + 2 T d_h ceil(T / B) + T and 'words materialised' is 4 T^2 + 4 T d_h; the last column is words tiled over the estimate 8 T^2 d_h^2 / M of the text.

| fast memory | d_h | B | floats in use | words tiled / words materialised | words tiled / (8 T^2 d_h^2 / M) |
|---|---:|---:|---:|---:|---:|
| 48 KiB | 64 | 41 | 12,177 | 0.78 | 1.17 |
| 48 KiB | 128 | 22 | 11,748 | 2.89 | 1.09 |
| 96 KiB | 64 | 74 | 24,420 | 0.43 | 1.31 |
| 96 KiB | 128 | 44 | 24,464 | 1.45 | 1.10 |
<!-- /worked -->

At $d_h = 64$ a workgroup with 48 KiB holds a block of 41 rows and moves 0.78 of the chain's words; with 96 KiB it moves 0.43. At $d_h = 128$ the same 48 KiB holds only 22 rows, and tiled attention moves 2.89 times the words of the chain, which is more, not less. Tiled attention is a saving of memory capacity first, since no $T \times T$ buffer exists, and a saving of traffic only when the fast memory holds enough rows for $2 d_h/B_q$ to fall below 4. Brain's kernels illustrate the point sharply, and a later section comes back to them.

The arithmetic is the other count, and it is nearly unchanged. The two products are the same $4T^2 d_h$ FLOPs. The softmax work moves around: the division by $\ell$ is done once per output element at the end and not once per score, which saves $T^2$ FLOPs, and the rescale costs $d_h + 4$ FLOPs per row and per tile.

<!-- worked: flash_attention.flop_counts -->
FLOPs of the forward pass of one head, T = 4096, d_h = 64, no mask. Materialised: 4 T^2 d_h for the two products plus 5 T^2 for the softmax (Chapter 7). Tiled: the same products, 4 T^2 for scaling, subtracting the maximum, the exponential and the addition to l, T d_h for the final division, and T ceil(T / B_k) (d_h + 4) for the rescales (subtract, exponential, multiply l, add, and d_h multiplies of the output).

| algorithm | B_k | FLOPs | relative to materialised |
|---|---:|---:|---:|
| materialised | - | 4,378,853,376 | 1.0000 |
| tiled | 8 | 4,504,944,640 | 1.0288 |
| tiled | 16 | 4,433,641,472 | 1.0125 |
| tiled | 64 | 4,380,164,096 | 1.0003 |
| tiled | 128 | 4,371,251,200 | 0.9983 |
<!-- /worked -->

With tiles of 8 keys, the rescale adds under 3 percent to the FLOPs of the chain, and with tiles of 64 it adds nothing visible; at 128 keys the saved divisions outweigh it. (The count is a lower bound in the convention of Volume IV, Chapter 3: the `exp` is counted as one operation, and a real unit spends several.) Tiled attention does the same arithmetic in fewer bytes and without the matrix, and the cost of the rescale is a reason to take tiles of at least a handful of keys, not an obstacle.

## One kernel, run on the simulator

A **flash attention** kernel is the tiled schedule and the online softmax in one kernel: exact attention, computed block by block, whose only outputs are the attention output and one number per row, the log-sum-exp $\mathrm{lse}_i = m_i + \ln \ell_i$ of its scaled scores. The second output costs $T$ numbers and is what the backward pass will need. The kernel in the book's pseudocode, one workgroup of $B_q$ invocations per block of query rows with one invocation per row (its query, $m$, $\ell$ and $\mathbf{o}$ are that invocation's registers):

```text
kernel flash_forward(Q: array<f32>[T, dh], K: array<f32>[T, dh], V: array<f32>[T, dh],
                     O: inout array<f32>[T, dh], LSE: inout array<f32>[T],
                     T: int, dh: int, B_q: int, B_k: int, causal: int) workgroup B_q:
    shared k_tile: array<f32>[B_k * dh]
    shared v_tile: array<f32>[B_k * dh]
    scale = 1 / sqrt(dh)
    i = w * B_q + l                          # the query row of this invocation
    q = zeros(f32, [dh])
    o = zeros(f32, [dh])
    if i < T:
        for c in 0 .. dh:
            q[c] = Q[i, c]
    m = -infinity
    ell = 0.0
    n_tiles = ceil(T / B_k)
    if causal == 1:                          # keys after the workgroup's last row are never needed
        n_tiles = min(n_tiles, min(w * B_q + B_q - 1, T - 1) // B_k + 1)
    for t in 0 .. n_tiles:
        for a in 0 .. ceil(B_k * dh / B_q):   # the invocations copy the tile together
            e = a * B_q + l
            if e < B_k * dh:
                j = t * B_k + e // dh
                k_tile[e] = K[j, e mod dh] if j < T else 0.0
                v_tile[e] = V[j, e mod dh] if j < T else 0.0
        barrier                              # the tiles are complete
        if i < T:
            s = zeros(f32, [B_k])
            for r in 0 .. B_k:
                j = t * B_k + r
                if j >= T or (causal == 1 and j > i):
                    s[r] = -infinity
                else:
                    s[r] = scale * dot(q, k_tile[r * dh : (r + 1) * dh])
            top = max(s)
            if top > -infinity:              # a tile wholly masked for this row changes nothing
                m_new = max(m, top)
                corr = exp(m - m_new)        # 0 on the first tile, where m = -infinity
                ell = ell * corr
                for c in 0 .. dh:
                    o[c] = o[c] * corr
                for r in 0 .. B_k:
                    pw = exp(s[r] - m_new)
                    ell = ell + pw
                    for c in 0 .. dh:
                        o[c] = o[c] + pw * v_tile[r * dh + c]
                m = m_new
        barrier                              # all are done reading before the next tile overwrites
    if i < T:
        for c in 0 .. dh:
            O[i, c] = o[c] / ell
        LSE[i] = m + log(ell)

launch flash_forward(Q, K, V, O, LSE, T, dh, B_q, B_k, 0) over ceil(T / B_q) workgroups
```

(`exp(-infinity)` is 0 and `max(x, -infinity)` is `x`; the pseudocode page says so.) The two barriers per tile have the jobs they had in Volume IV, Chapter 6: the first completes the tile before anyone reads it, the second keeps a fast invocation from overwriting a tile a slow one still reads. Invocations whose row is past the end of the matrix still reach both, so the barrier rule of Volume IV, Chapter 5 holds.

Run it on the simulator of Volume IV, Chapter 5, on a case whose sizes divide nothing, 11 rows in blocks of 4 and tiles of 3, so that every edge is ragged, and compare with materialised attention:

<!-- worked: flash_attention.simulated_forward -->
Generator: the book's SplitMix64, seed 0x65487e58ddc4fb27 (`seed_of` of the name `flash_attention.simulated`). T = 11 rows, d_h = 4, query blocks of B_q = 4 rows (one workgroup of 4 invocations each) and key tiles of B_k = 3 rows, so 3 workgroups and 4 key tiles, neither a divisor of T. Draws, in this order: Q, K, V (each 11 by 4, row by row, `normal()`). The kernel runs on the simulator of Volume IV, Chapter 5, in double precision (scale 1 / sqrt(d_h)); the key tile and the value tile are shared memory (24 words), the query row, m, l and the output row of each invocation are registers. Loads and stores count numbers (words) moved to or from the buffers Q, K, V, O and the log-sum-exp; the loads equal T d_h plus the key and value rows each workgroup loads, counted from the tile geometry. The largest buffer of the kernel has 44 words, against T^2 = 121.

| mask | workgroups | key tiles visited | global loads | global stores | O differs by less than | log-sum-exp differs by less than |
|---|---:|---:|---:|---:|---:|---:|
| no mask | 3 | 12 of 12 | 308 | 55 | 1e-12 | 1e-12 |
| causal | 3 | 9 of 12 | 252 | 55 | 1e-12 | 1e-12 |
<!-- /worked -->

The kernel's output and its log-sum-exp match the materialised attention to rounding error, with and without the mask, and the loads are the count of the formula: $T d_h$ for the queries plus the keys and values each workgroup loads. The largest buffer the kernel ever touches has 44 words, the size of $Q$, against the $T^2 = 121$ words the score matrix would take; at the sizes of the opening that gap is the 128-fold ratio of the first table.

## Skipping the masked half

For a language model the mask is causal, and a tile either is entirely visible, entirely masked, or cut by the diagonal. The first kind needs no mask; the second need not be loaded at all; only the third needs the per-entry test that the kernel above has in its inner loop. Classify the tiles of a score matrix for $T = 24$ in blocks of 4 query rows against tiles of 4 keys, and in blocks of 8 rows against tiles of 4:

![Tiles of a causal score matrix at T = 24 for two blockings, query blocks down the side and key tiles across: with 4 rows by 4 keys there are 36 tiles, 15 full, 6 on the diagonal and 15 skipped; with 8 rows by 4 keys there are 18 tiles, 6 full, 6 diagonal and 6 skipped. Skipped tiles are never loaded; full tiles are computed without a mask; diagonal tiles are computed and the keys past each row are masked.](img/causal-tile-grid.svg)

<!-- worked: flash_attention.causal_tiles -->
Tiles of the score matrix under the causal mask. A tile is the set of scores of B_q consecutive query rows against B_k consecutive keys: skipped when every key is after every row of the block, full when every key is at or before the first row, diagonal otherwise (the last block of a ragged T is smaller). The scores the kernel computes are those of the tiles that are not skipped, counted entry by entry as a fraction of T^2; the mask lets (T + 1) / (2 T) of them through (0.5208 at T = 24, 0.5001 at T = 4096).

| T | B_q x B_k | tiles | full | diagonal | skipped | scores computed / T^2 |
|---|---:|---:|---:|---:|---:|---:|
| 24 | 4 x 4 | 36 | 15 | 6 | 15 | 0.5833 |
| 24 | 8 x 4 | 18 | 6 | 6 | 6 | 0.6667 |
| 4,096 | 64 x 64 | 4,096 | 2,016 | 64 | 2,016 | 0.5078 |
| 4,096 | 64 x 8 | 32,768 | 16,128 | 512 | 16,128 | 0.5078 |
| 32,768 | 64 x 8 | 2,097,152 | 1,046,528 | 4,096 | 1,046,528 | 0.5010 |
<!-- /worked -->

The saving tends to one half, from above: the diagonal tiles are computed in full, so a coarse tile computes more scores than the mask allows. At $T = 4096$ the kernel's shape (64 rows by 8 keys) computes 0.5078 of the $T^2$ scores where the mask permits 0.5001, and at $T = 32{,}768$ it computes 0.5010 against 0.5000. The block shape matters at short lengths: 8 rows against 4 keys at $T = 24$ computes two thirds of $T^2$. A kernel that skips masked tiles also loads fewer keys and values, which the simulator of the previous section counted: 252 loads against 308 at $T = 11$.

The diagonal tiles hide a hazard that has nothing to do with the arithmetic of the mask. Within a block of rows, the rows are cut at different places: a tile can be entirely masked for the first row of the block and partly visible for the last. A row whose tile is wholly masked has every score at $-\infty$, and if the row has seen nothing before, the update forms $m - m'$ with both infinite, which is not a number. Take one row of four keys in two tiles of two that sees only the second tile, so that its first tile is masked entirely, and run the update three ways against the exact softmax of the visible keys:

<!-- worked: flash_attention.masked_first_tile -->
One row of four keys in two tiles of two; the scores are (0.5, 1.0, 2.0, -1.0) and the row sees only keys 2 and 3, so its first tile is masked entirely. The values are (1.0, 2.0), (3.0, -1.0), (0.0, 1.5), (2.0, 2.0). Output of the row from the exact softmax of its visible keys and from three ways of running the online update (double precision):

| method | output | differs from the exact softmax by less than |
|---|---|---|
| the softmax over the visible keys only | (0.094852, 1.523713) | - |
| minus infinity, no test of the tile | (nan, nan) | nan from -inf minus -inf |
| a finite sentinel -3.4e38, no test | (0.094852, 1.523713) | 1e-12 |
| minus infinity, tile skipped when all of it is masked | (0.094852, 1.523713) | 1e-12 |

A row that sees none of the four keys, with the sentinel and no test, ends with l = 4.0 and the output (1.500000, 1.125000), the plain mean (1.500000, 1.125000) of the four values.
<!-- /worked -->

Running the update with true $-\infty$ and no test gives not-a-number, as $-\infty - (-\infty)$ must (the second row). The test in the kernel above, "skip a tile with no visible score", is the cure (the fourth row). The third row is a finite sentinel, the most negative float of 32 bits in place of $-\infty$, with no test at all, which is what brain's kernels use, and it is correct for a reason worth stating: on a masked first tile the update treats the masked scores as real, gives them weight $e^0 = 1$, and so adds garbage to $\ell$ and $\mathbf{o}$, but the first tile with a visible score raises the maximum to a finite value and the correction $e^{-3.4 \times 10^{38} - m'}$ is exactly 0, which wipes the garbage clean. The row is right as long as it sees at least one key. A row that sees none gets the plain mean of the values, as the last line of the block shows, instead of the zero or the error that a reader of the output would hope for. In the causal kernels this cannot happen: key 0 is visible to every row, and it sits in the first tile.

## The backward pass without the matrix

Training needs gradients, and Chapter 7's backward pass reads $\mathbf{P}$, the matrix that tiled attention has been careful never to write. There are two ways to give the backward pass its weights: keep $\mathbf{P}$, which brings back the memory, or rebuild each weight from what is cheaper to keep. The forward pass saved the output $\mathbf{O}$ (the next layer needs it anyway) and one number per row, the log-sum-exp. From them any weight is

$$
p_{ij} = \exp\big(s_{ij} - \mathrm{lse}_i\big), \qquad \mathrm{lse}_i = m_i + \ln \ell_i,
$$

since $e^{s - m}/\ell = e^{s - m - \ln \ell}$. Recomputing $s_{ij} = \mathbf{q}_i \cdot \mathbf{k}_j/\sqrt{d_h}$ for the tile at hand costs one matrix product, and the weight costs one exponential. Saving the row's maximum $m_i$ alone is not enough, as the check below shows: $\ell_i$ is needed too, and the log-sum-exp holds both in one number. The **flash attention backward pass** is the backward pass of Chapter 7 with $\mathbf{P}$ recomputed tile by tile from $\mathrm{lse}$, and it stores $T$ numbers per head where Chapter 7's stored $T^2$.

One piece of Chapter 7's derivation needs a replacement. The softmax rule is $dS_{ij} = p_{ij}(dp_{ij} - c_i)$ with the row term $c_i = \sum_j p_{ij}\,dp_{ij}$, a sum over the whole row of the matrix, which a tile does not have. But $dp_{ij} = d\mathbf{o}_i \cdot \mathbf{v}_j$ (the gradient of the weight is the gradient of the output dotted with the value it multiplies), so

$$
c_i = \sum_j p_{ij}\,(d\mathbf{o}_i \cdot \mathbf{v}_j) = d\mathbf{o}_i \cdot \Big(\sum_j p_{ij}\,\mathbf{v}_j\Big) = d\mathbf{o}_i \cdot \mathbf{o}_i =: D_i .
$$

The sum over the row collapses into a dot product of two vectors of length $d_h$: the incoming gradient with the output the forward pass already saved. $D_i$ costs one pass over $d\mathbf{O}$ and $\mathbf{O}$, $T d_h$ products, and is computed before any tile. Everything else is Chapter 7's chain with $\mathbf{P}$ and $D_i$ standing where it had $\mathbf{P}$ and $c_i$:

$$
\begin{aligned}
d\mathbf{V} &= \mathbf{P}^\top d\mathbf{O}, & d\mathbf{P} &= d\mathbf{O}\,\mathbf{V}^\top, \\
dS_{ij} &= p_{ij}\,(dp_{ij} - D_i), \\
d\mathbf{Q} &= \tfrac{1}{\sqrt{d_h}}\,d\mathbf{S}\,\mathbf{K}, & d\mathbf{K} &= \tfrac{1}{\sqrt{d_h}}\,d\mathbf{S}^\top \mathbf{Q}.
\end{aligned}
$$

Each tile $(i\text{-block}, j\text{-tile})$ contributes to three outputs: $d\mathbf{Q}$ of its rows, $d\mathbf{K}$ and $d\mathbf{V}$ of its keys. A tile loop that owns the rows accumulates $d\mathbf{Q}$ in registers and has to send $d\mathbf{K}$ and $d\mathbf{V}$ somewhere, and one that owns the keys has the opposite problem. Without atomics the clean solution is two passes, each owning what it accumulates: a query-side pass over the key tiles that produces $d\mathbf{Q}$, and a key-side pass over the query blocks that produces $d\mathbf{K}$ and $d\mathbf{V}$, each recomputing the scores it needs:

```text
function tile_weights(Qb: array<real>[nr, dh], Kb: array<real>[nc, dh], lse_b: array<real>[nr],
                      r0: int, c0: int, causal: bool) -> array<real>[nr, nc]:
    S = matmul(Qb, transpose(Kb))
    P = zeros(real, [nr, nc])
    for a in 0 .. nr:
        for b in 0 .. nc:
            if not (causal and c0 + b > r0 + a):
                P[a, b] = exp(S[a, b] / sqrt(dh) - lse_b[a])
    return P

function flash_backward(Q: array<real>[T, dh], K: array<real>[T, dh], V: array<real>[T, dh],
                        O: array<real>[T, dh], lse: array<real>[T], dO: array<real>[T, dh],
                        B_q: int, B_k: int, causal: bool)
        -> (array<real>[T, dh], array<real>[T, dh], array<real>[T, dh]):
    D = [dot(dO[i, :], O[i, :]) for i in 0 .. T]
    dQ = zeros(real, [T, dh])
    dK = zeros(real, [T, dh])
    dV = zeros(real, [T, dh])
    for qb in 0 .. ceil(T / B_q):                       # query side: one block of rows at a time
        r0 = qb * B_q
        r1 = min(r0 + B_q, T)
        for kb in 0 .. ceil(T / B_k):
            c0 = kb * B_k
            c1 = min(c0 + B_k, T)
            if causal and c0 > r1 - 1:
                continue                                # every key of the tile is after every row
            P = tile_weights(Q[r0:r1, :], K[c0:c1, :], lse[r0:r1], r0, c0, causal)
            dP = matmul(dO[r0:r1, :], transpose(V[c0:c1, :]))
            dS = zeros(real, [r1 - r0, c1 - c0])
            for a in 0 .. r1 - r0:
                for b in 0 .. c1 - c0:
                    dS[a, b] = P[a, b] * (dP[a, b] - D[r0 + a]) / sqrt(dh)
            dQ[r0:r1, :] = dQ[r0:r1, :] + matmul(dS, K[c0:c1, :])
    for kb in 0 .. ceil(T / B_k):                       # key side: one tile of keys at a time
        c0 = kb * B_k
        c1 = min(c0 + B_k, T)
        for qb in 0 .. ceil(T / B_q):
            r0 = qb * B_q
            r1 = min(r0 + B_q, T)
            if causal and r1 - 1 < c0:
                continue                                # every row of the block is before every key
            P = tile_weights(Q[r0:r1, :], K[c0:c1, :], lse[r0:r1], r0, c0, causal)
            dP = matmul(dO[r0:r1, :], transpose(V[c0:c1, :]))
            dS = zeros(real, [r1 - r0, c1 - c0])
            for a in 0 .. r1 - r0:
                for b in 0 .. c1 - c0:
                    dS[a, b] = P[a, b] * (dP[a, b] - D[r0 + a]) / sqrt(dh)
            dV[c0:c1, :] = dV[c0:c1, :] + matmul(transpose(P), dO[r0:r1, :])
            dK[c0:c1, :] = dK[c0:c1, :] + matmul(transpose(dS), Q[r0:r1, :])
    return (dQ, dK, dV)
```

Nothing larger than $B_q \times B_k$ is ever formed, and $D$ is a vector of $T$ numbers. A derivation of this length is the kind that Chapter 7 checked in two ways, against the formulas it implements and against central differences, and this one gets the same treatment, on the same seed machinery: tiled backward against the materialised backward of Chapter 7, and the tiled forward and backward together against central differences of a loss. The last column checks the identity $D_i = \sum_j p_{ij}\,dp_{ij}$ on the same rows.

<!-- worked: flash_attention.backward_check -->
Generator: the book's SplitMix64, seed 0x419b36474ce369e6 (`seed_of` of the name `flash_attention.backward`). T = 7, d_h = 3, query blocks of 3 rows and key tiles of 2 rows. Draws, in this order: Q, K, V and the incoming gradient dO (each 7 by 3, row by row, `normal()`). The loss is the sum over all entries of dO times O, so the gradient of the loss with respect to O is dO. 'Tiled backward' recomputes every weight tile by tile from the saved O and log-sum-exp; 'materialised' is the backward of Chapter 7. Central differences have step 1e-05, are taken through the tiled forward on every entry of the tensor, and are compared as |a - n| / max(|a|, |n|) over the entries whose gradient is above 1e-4. The last column is the largest |dO_i . O_i - sum_j P_ij dP_ij| over the rows. Every cell is the power of ten the difference stays below.

| mask | tensor | tiled vs materialised, absolute | tiled vs central differences, relative | D_i identity, absolute |
|---|---|---:|---:|---:|
| no mask | dQ | 1e-12 | 1e-8 | 1e-12 |
| no mask | dK | 1e-12 | 1e-9 | 1e-12 |
| no mask | dV | 1e-12 | 1e-9 | 1e-12 |
| causal | dQ | 1e-12 | 1e-8 | 1e-12 |
| causal | dK | 1e-12 | 1e-9 | 1e-12 |
| causal | dV | 1e-12 | 1e-9 | 1e-12 |
<!-- /worked -->

The tiled gradients agree with the materialised ones to rounding error and with central differences to a relative $10^{-8}$ or $10^{-9}$, with and without the mask, at sizes (7 rows, blocks of 3, tiles of 2) where nothing divides anything, and the identity of $D_i$ holds on every row. A check that passes should also be able to fail, as in Chapter 7, and the places where a flash backward pass goes wrong are specific: the term $D_i$, the saved statistic, and the mask in the recomputed scores.

<!-- worked: flash_attention.wrong_backward_passes -->
The check of the previous block (same inputs, same tiles) run on the correct tiled backward pass and on deliberately wrong ones. Cells are the largest relative difference |a - b| / max(|a|, |b|) of the tensor from the materialised backward pass over the entries whose gradient is above 1e-4; a difference of 1e-6 or more is written to one decimal, a smaller one as the power of ten it stays below. Dropping D_i leaves dV, which does not use it, correct.

| mask | backward pass | dQ | dK | dV |
|---|---|---:|---:|---:|
| no mask | the chapter's backward pass | 1e-12 | 1e-12 | 1e-12 |
| no mask | D_i left out (taken as 0) | about 1.1 | about 1.6 | 1e-12 |
| no mask | the running maximum m_i saved in place of the log-sum-exp | about 0.8 | about 2.0 | about 1.7 |
| causal | the chapter's backward pass | 1e-12 | 1e-12 | 1e-12 |
| causal | D_i left out (taken as 0) | about 1.6 | about 1.3 | 1e-12 |
| causal | the running maximum m_i saved in place of the log-sum-exp | about 0.6 | about 0.8 | about 1.8 |
| causal | the causal mask left out of the recomputed scores | about 1.3 | about 1.6 | about 1.6 |
<!-- /worked -->

Leaving out $D_i$ spares $d\mathbf{V}$, which uses $\mathbf{P}$ and $d\mathbf{O}$ and not $d\mathbf{S}$, and corrupts $d\mathbf{Q}$ and $d\mathbf{K}$, the pattern that told Chapter 7's reader which step was missing. Saving the maximum $m_i$ in place of the log-sum-exp makes every weight too large by the factor $\ell_i$ and corrupts all three. And in the causal case a backward pass that forgets to mask its recomputed scores weights keys that the forward pass never saw, which is the same fault as the leaked future of Chapter 7's mask, now in a place that the forward pass's own tests do not reach: the forward is right, and only the gradient is wrong.

The price of recomputation is arithmetic. Count the products of one head in units of one $T \times T \times d_h$ product ($2T^2 d_h$ FLOPs):

<!-- worked: flash_attention.backward_cost -->
Products of the backward pass of one head, counted in units of one T x T x d_h product (2 T^2 d_h FLOPs, no mask). The last row was counted by running the tiled backward of the previous blocks on a case with T = 64, d_h = 8 and tiles of 16 and counting the multiply-adds of every tile product: 229,376 multiply-adds, which is 7 T^2 d_h = 229,376. The forward pass is 2 such products (S and P V), so a training step spends 2 + 4 = 6 on the materialised path and 2 + 5 = 7 or 2 + 7 = 9 when scores are recomputed.

| backward pass | products | FLOPs at T = 64, d_h = 8 | relative to materialised |
|---|---:|---:|---:|
| materialised (Chapter 7): dP, dV, dQ, dK | 4 | 262,144 | 1.00 |
| S recomputed once, one pass: S, dP, dV, dK, dQ | 5 | 327,680 | 1.25 |
| two passes: (S, dP, dQ) and (S, dP, dV, dK) | 7 | 458,752 | 1.75 |
<!-- /worked -->

A single pass that recomputed $S$ once would need five products where the materialised backward pass needs four, but five requires the three outputs to be accumulated in one loop and so atomics or a reduction over blocks; the two-pass form recomputes $S$ and $d\mathbf{P}$ twice and costs seven, the price of writing each output exactly once. Whether recomputing is worth it is the recompute-or-store question of Volume IV, Chapter 8, and here it is easy: the alternative is $T^2$ numbers per head for each of $\mathbf{P}$ and $d\mathbf{S}$, and the extra products are done at the high arithmetic intensity of a tile, from numbers already on chip. Dao (2023) refined the schedule of the original method, the order of its loops and the division of work between workgroups, without changing the algebra derived here.

## One query against a long cache

All of the above gets its parallelism from the query rows: each workgroup owns a block of them, and a long sequence has thousands. Generation (Chapter 12) is the opposite shape. A decode step has one query per sequence and head, and the keys are the whole cache. The grid of a query-blocked kernel is $\lceil 1/B_q \rceil = 1$ workgroup per head, so a model with 16 heads decoding one sequence launches 16 workgroups on a device that has dozens of compute units, and each of the 16 walks the cache alone, tile after tile, with the rest of the device idle.

The parallelism is in the keys, and the merge defined early in this chapter is what makes it available. Cut the cache into $S$ contiguous ranges. Give each range to its own workgroup, which computes the state $(m_s, \ell_s, \mathbf{o}_s)$ of its keys by the kernel above (without the final division) and writes it. A second step merges the $S$ states, which the merge allows in any order, and divides: $\mathbf{O} = \mathbf{o}/\ell$ for the merged state, with

$$
\begin{aligned}
m &= \max_s m_s, \\
\ell &= \sum_s \ell_s\,e^{m_s - m}, \\
\mathbf{o} &= \sum_s \mathbf{o}_s\,e^{m_s - m}.
\end{aligned}
$$

A range that lies beyond the end of the cache (the grid is the same for every sequence of a batch, and a short sequence leaves ranges empty) returns the empty state $(-\infty, 0, \mathbf{0})$, which the merge treats as nothing. Test it on caches of the lengths the real kernel's test uses, which straddle the tile boundary on purpose, and with the mistake that a combine step is most likely to contain, the factor $e^{m_s - m}$ left out of the sum of the normalizers:

<!-- worked: flash_attention.decode_split -->
Generator: the book's SplitMix64, seed 0xc4033ac94daca479 (`seed_of` of the name `flash_attention.decode`). One query of width d_h = 4 against a cache of up to 100 rows (K and V each 100 by 4). Draws, in this order: the query (`normal()` 4 times), then K and V row by row, `normal()`; a shorter cache is the first rows. Keys are read in tiles of 8, each of the 7 workgroups walks 2 consecutive tiles and returns its state (m, l, o); the same number of workgroups is dispatched for every length, and one whose tiles lie beyond the cache returns the empty state (m = -infinity, l = 0, o = 0). The combine merges the states left to right (or as a balanced tree) and divides o by l. The last column is the combine with exp(m_i - m) left out of the sum l: its largest absolute error, to three decimals.

| cache length | tiles of 8 | workgroups with keys | empty workgroups | combine differs from attention by less than | tree combine differs by less than | combine that drops the factor, error |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 1 | 6 | 1e-12 | 1e-12 | 0.000 |
| 7 | 1 | 1 | 6 | 1e-12 | 1e-12 | 0.000 |
| 8 | 1 | 1 | 6 | 1e-12 | 1e-12 | 0.000 |
| 9 | 2 | 1 | 6 | 1e-12 | 1e-12 | 0.000 |
| 41 | 6 | 3 | 4 | 1e-12 | 1e-12 | 0.247 |
| 100 | 13 | 7 | 0 | 1e-12 | 1e-12 | 0.100 |
<!-- /worked -->

The split and combined result is the attention of the whole cache to rounding error for every length, in a loop or in a balanced tree, with empty ranges as well as full ones. The mistaken combine is right for lengths up to 9 (one range holds all the keys, so the factor it omits multiplies the $\ell$ of an empty range, which is 0) and wrong as soon as two ranges hold keys with different maxima, which is why a test with one range per sequence would never find it.

What the split adds in traffic is small. The cache must be read once whatever the grid, and the states cost $2S(d_h + 2)$ numbers per head, written by the first step and read by the second:

<!-- worked: flash_attention.decode_traffic -->
One decode step of one sequence: a cache of T rows of width d_h = 128 for K and for V in float32, split across S workgroups per head. Each split writes its state (m, l and d numbers of o) and the combine reads it back, so the extra traffic is 2 S (d_h + 2) numbers per head; the last column is the number of workgroups for a model with 16 heads and one sequence.

| T | S | cache bytes read per head | partial states, bytes written and read | partial / cache | workgroups, 16 heads |
|---|---:|---:|---:|---:|---:|
| 4,096 | 1 | 4,194,304 | 0 | 0.00000 | 16 |
| 4,096 | 8 | 4,194,304 | 8,320 | 0.00198 | 128 |
| 4,096 | 32 | 4,194,304 | 33,280 | 0.00793 | 512 |
| 32,768 | 1 | 33,554,432 | 0 | 0.00000 | 16 |
| 32,768 | 8 | 33,554,432 | 8,320 | 0.00025 | 128 |
| 32,768 | 32 | 33,554,432 | 33,280 | 0.00099 | 512 |
<!-- /worked -->

At 32,768 keys and 32 ranges the states are a thousandth of the cache, and the grid of a 16-head model grows from 16 to 512 workgroups. This two-step structure is the usual one for decode kernels that split over keys. Brain's kernel headers record that they derived the merge from the kernel's own per-tile update, which is the same identity applied one level down.

## Two exact answers that are not the same bits

Both the tiled and the materialised attention are exact in the sense that they compute the same real numbers. They do not compute the same floating-point numbers. The tiled version sums the exponentials in tiles, rescales in between, and divides the weighted sum once by $\ell$ at the end where the materialised version divides every weight first. The order of the additions differs, and Chapter 7 of Volume IV showed what that means: each addition rounds, the roundings depend on the size of the partial sums, and a different order gives a different last bit. The test of a flash kernel is therefore a tolerance and never an identity.

Here are the same causal attention, from one set of float32 inputs, computed in float32 one operation at a time, materialised and tiled with several tile sizes:

<!-- worked: flash_attention.float32_numerics -->
Generator: the book's SplitMix64, seed 0xc3cf4aee0561a563 (`seed_of` of the name `flash_attention.float32`). Causal attention of one head, T = 48, d_h = 16. Draws, in this order: Q, K, V (each 48 by 16, row by row, `normal()` rounded to float32). Every product, sum, difference and quotient is rounded to float32 one operation at a time (no fused multiply-add), sums run left to right (a dot product over the head width, the normalizer, and the weighted sum over the keys), exp is the library exp of double precision rounded to float32, and the scale is the float32 nearest to 1 / sqrt(d). The tiled version applies the update of the chapter per tile of B_k keys and divides by l at the end; the materialised version divides each weight first. The reference in double is attention in double precision from the same float32 inputs. Differences are the largest absolute difference over the 768 outputs, in units of u = 2^-24 (the float32 unit roundoff) and rounded up to a whole number; the outputs are below 3 in magnitude. The last column counts outputs equal to the materialised float32 result in every bit; the count depends on the library exp, which is why the page names it.

| version | difference from materialised float32, units of u | difference from double precision, units of u | outputs bit-identical to materialised |
|---|---:|---:|---:|
| materialised | - | 5 | 768 of 768 |
| tiled, B_k = 1 | 6 | 5 | 186 of 768 |
| tiled, B_k = 2 | 6 | 5 | 168 of 768 |
| tiled, B_k = 4 | 6 | 5 | 148 of 768 |
| tiled, B_k = 8 | 6 | 5 | 158 of 768 |
| tiled, B_k = 16 | 6 | 5 | 176 of 768 |
| tiled, B_k = 48 | 6 | 5 | 186 of 768 |
<!-- /worked -->

Whatever the tile size, a few units of the float32 roundoff separate the two versions from each other and from the double-precision answer, which is what two correct but differently ordered computations look like. And only about a fifth of the 768 outputs agree with the materialised result in every bit, for every tile size including the one that makes a single tile of the whole row: even there the division order differs. This is the reason a flash kernel is gated on a tolerance against the materialised chain, and the reason a model's output may change in its last digits when the sequence length crosses the threshold at which the engine switches from one path to the other.

## Brain's flash kernels

Brain has a family of these kernels, and the useful way to read them is as this chapter's loop with the constraints of a particular machine. The causal forward is `crates/kernels/wgsl/flash_attn_causal_gqa.wgsl`; its two backward kernels are `flash_attn_causal_gqa_bwd_dq.wgsl` and `flash_attn_causal_gqa_bwd_dkv.wgsl`, the query-side and key-side passes of the section above. They share the grouped-query layout (a block of query heads sharing a key head, a later chapter's subject), which affects only where a key tile is read from. `crates/model/src/block.rs` builds them: `flash_gqa_causal_fwd` dispatches the forward and returns the log-sum-exp, and `flash_gqa_causal_bwd` runs the two backward kernels, with a scratch vector $D_i$ passed between them.

The kernel does not give one invocation one query row, as the pseudocode did. A workgroup has 256 invocations, and the head width is split across *lanes*:

<!-- worked: flash_attention.kernel_geometry -->
The shape of the causal forward kernel of brain for the three head widths it serves: 256 invocations per workgroup, key tiles of B_k = 8 rows, 32 channels per invocation. lanes = 4 up to head width 128 and 8 above; a workgroup covers 256 / lanes query rows. 'q and o per invocation' is 2 x 32 registers. 'If one invocation owned a row' is 2 d_h, to be compared with the 255 registers per invocation that the kernel's own header names for the device it was written for. Shared KiB is the key tile and the value tile (2 x 8 x lanes x 32 floats) plus the partial dot products (8 x 256 floats).

| head width | lanes | query rows per workgroup | q and o per invocation | if one invocation owned a row | shared KiB used |
|---|---:|---:|---:|---:|---:|
| 64 | 4 | 64 | 64 | 128 | 16 |
| 128 | 4 | 64 | 64 | 256 (over 255) | 16 |
| 256 | 8 | 32 | 64 | 512 (over 255) | 24 |
<!-- /worked -->

The reason is registers. One invocation holding a row of a head of width 128 needs the 128 numbers of its query and the 128 of its output accumulator, 256 values, more than the 255 registers per invocation that the kernel's header names for the Pascal-generation card it was written for; the spilled arrays live in local memory, which is global memory, and the inner loop then does about three local accesses per two FLOPs. The fix is to give each row to 4 invocations (8 above head width 128) that hold 32 channels each: 64 registers for $\mathbf{q}$ and $\mathbf{o}$, a compile-time loop count, and a dot product that each lane computes in part and the lanes sum through a small buffer in shared memory, once per tile. The tile is 8 keys, so the three barriers of a tile are the staging of the keys and values, the partial dot products, and the end of the tile. There are no subgroup shuffles in this kernel, which is deliberate: the same source runs on every device that can run barriers and a workgroup of 256 with enough shared memory, and a faster path for devices with tensor cores is a different, registered kernel: `FlashGqaIds::native` offers the native kernels of the CUDA backend first on a device that has them, and setting it false pins the portable fp32 kernels, which is what gradient checks and parity gates are stated against. The sum across the lanes is a reduction of Volume IV's kind with four terms, and it is one of the places where the order of the additions differs from the materialised chain's.

The counting of this chapter says what that design buys. A workgroup holds $B_q = 64$ rows, or 32 above head width 128, and the traffic is $2T^2 d_h / B_q$ words per head: at $d_h = 128$ that is $4T^2$, the number of words of the three-kernel chain's lower bound. Brain's flash kernel on that device saves the *capacity* of the matrix, and the traffic only matches. The commits that introduced flash attention to the repository bear this out. Commit `c655f60f6` (July 2026) added `flash_attn_bidir.wgsl`, a fused kernel for a diffusion model's bidirectional attention, whose message gives the motive (peak attention memory $O(T \cdot \text{head dim})$ instead of $O(T^2)$, so that high-resolution latents run past the per-buffer limit) and the check (with flash forced on the real model still matches its reference). Commit `d538e95ba`, the same day, moved the query and accumulator into registers to raise the number of resident workgroups, and added the heuristic that follows from the count: the materialised path uses a tuned register-tiled matrix product, where it fits it beats the hand-written fused loops on a card without tensor cores, so flash is "a MEMORY escape hatch, not a blanket speedup", to be switched on automatically only when the score buffer approaches the device's single-buffer limit.

The causal kernel came from a limit of that kind in August 2026, and from a mistake that the count above predicts. Commit `3837fe359` records the problem: a multimodal model's decoder was given a prompt of 7,484 tokens by a long system prompt and tool schemas, the score buffer at that length is 6.68 GiB (what 32 heads of float32 scores come to, the first row below), and the device reported an out-of-memory error.

<!-- worked: flash_attention.score_slabs -->
Score slabs in float32, H heads and T rows: 4 H T^2 bytes. The second row is the smallest T for which the slab of 16 heads exceeds 2^31 - 1 bytes, the largest storage-binding size the device of that history allowed.

| H | T | slab, bytes | GiB |
|---|---:|---:|---:|
| 32 | 7,484 | 7,169,312,768 | 6.68 |
| 16 | 5,793 | 2,147,766,336 | 2.00 |
| 24 | 16,384 | 25,769,803,776 | 24.00 |
| 24 | 32,768 | 103,079,215,104 | 96.00 |
<!-- /worked -->

<!-- perf-number: historical measurement, commit 3837fe359 -->
The first version of the flash kernel copied the layout of the bidirectional one, a row per invocation, and the message reports it correct and about 300 times too slow, measured for one decoder layer at that length on a Tesla P40; moving to the lane split of the table above, the kernel's header explains, was the whole cure, and the message reports the result equal to the materialised path to 3e-7. The register count is the whole story and it is a count, not a measurement: 256 values against a limit of 255.

## What the trio is tested against, and what it made possible

Training is where the backward kernels matter. Commit `12901f738` (September 2026) switched the training attention of a language model's fine-tuning to the three flash kernels, and knowledge #193 records why: the materialised chain keeps three slabs per layer, and with 16 heads one slab passes the card's single-buffer limit (2047 MiB as the device reports it) at 5,793 tokens, the second row of the table above. A 5,824-token step failed at the creation of the scores buffer.

<!-- perf-number: historical measurement, knowledge #193 -->
Measured in September 2026 on one Tesla P40, a LoRA fine-tuning step of a 0.6-billion-parameter model at 16,384 tokens used 11,584 MiB of device memory for the whole process with the flash kernels, where one of the materialised path's three slabs alone would be 16 GiB at that length and the three together 48.

Commit `436365bb5` (October 2026) widened the trio from head width 128 to 256, the width of the full-attention layers of a larger model, by letting the lane count follow the head, and replaced two single-case tests with `crates/model/tests/it/flash_gqa_train.rs`. That file is the chapter's checks, run on a device: the flash forward, its log-sum-exp and its three gradients are compared with the materialised chain at head widths 40, 128 and 256, with grouping ratios of up to six query heads per key head, a batch of two and lengths that no tile divides, within $10^{-4}$ of the largest magnitude; and the whole mixer is gradient-checked by finite differences at distinct dimensions, so that a swapped stride cannot cancel (knowledge #4), with a count that at least 24 of its 48 sampled gradients are not trivial, so that the check cannot pass on a backward pass that returns zeros. The same kernels, with the mask and the early exit removed by a specialization constant, serve bidirectional attention (commit `ce1456612`, October 2026, checked in `crates/model/tests/it/flash_attn_bidir_train.rs` against a double-precision reference that materialises the whole $T \times T$ matrix).

The attention buffers in this design are linear in $T$. For the 24-head, 4-key-head, width-256 layer of that commit at 16,384 tokens:

<!-- worked: flash_attention.flash_buffers -->
The attention buffers of one training layer of a model with 24 query heads, 4 key-value heads and head width 256 at T = 16384, float32. Tiled path: q, k, v, ctx and the log-sum-exp for the forward, d_ctx, d_q, d_k, d_v and D_i for the backward. Materialised path: three slabs of 4 H T^2 bytes (scores, probabilities, score gradients).

| buffers | bytes | GiB |
|---|---:|---:|
| q and ctx (each) | 402,653,184 | 0.375 |
| k and v (each) | 67,108,864 | 0.062 |
| log-sum-exp (and D_i) | 1,572,864 | 1.5 MiB |
| tiled forward | 941,096,960 | 0.88 |
| tiled forward and backward | 1,882,193,920 | 1.75 |
| one slab | 25,769,803,776 | 24.0 |
| three slabs | 77,309,411,328 | 72.0 |
<!-- /worked -->

Knowledge #240 reports a measured growth of 6.5 GiB of card memory for the whole layer at that length, inputs and the rest of the mixer included, about a quarter of one materialised slab, and a test (`real_width_flash_mixer_on_cuda`) that fails if the growth passes 14 GiB, because "memory is not O(T)". The measurement is one device on one day, and the quarter is arithmetic on it; the 1.75 GiB is arithmetic on the buffers.

The decode side followed the same logic. Commit `f9f495bb0` (September 2026) added `paged_flash_decode_split.wgsl` and `paged_flash_decode_combine.wgsl`, the two steps of the previous section, with the cache addressed through a table of block numbers (a later chapter's paged cache; it changes where a key is read from and nothing else). The split kernel's range is a runtime parameter, `tiles_per_split`, which changes the trip count of its loop and neither its shared memory nor its code. The combine kernel is one invocation per output channel, a serial loop over the ranges applying the two-term merge, and its header states the empty-range convention: a range past the cache writes $m = -3.4 \times 10^{38}$, $\ell = 0$ and $\mathbf{o} = \mathbf{0}$, and the factor for it evaluates to 0 against any finite maximum. The kernel-level test `paged_flash_decode_split_matches_batched_triad` in `crates/model/src/paged.rs` uses the lengths 1, 7, 8, 9, 41 and 100 of the table above, with tiles of 8 keys and two tiles per range, so that 100 keys need 7 ranges and the short sequences leave six empty. The milestone's ledger records its mutation check: removing the factor from the update of $\ell$ in the combine kernel made the test fail at once, with a maximum error of 0.765 against 2.4e-7 for the correct kernel, which is the experiment of the table's last column.

The commit also records a result the arithmetic cannot give. On the Intel integrated GPU it was profiled on, the split kernel beat the three-kernel chain reliably only with one sequence in the batch, was mixed with two and lost with four or more, because a batch of sequences already fills the device with independent workgroups. The selector in `crates/backend-api/src/select.rs` encodes it: the fused decode is offered for the 32-bit cache only when the batch is one. The split is a remedy for an idle device, and when the device is not idle it only adds the second step.

One more lesson from the same files is about naming, not attention. Knowledge #352 records that the three flash kernels, specialized for the bidirectional case, passed every debug-build test and then failed in the release build with a mismatch of binding counts, because the code had recovered each kernel's base name by comparing the address of its source string, and, as the entry infers, an optimized build gave all three the same name. A name should be stated beside its source and not derived from where the source happens to sit.

## Check your understanding

1. A student implements the online softmax of this chapter but, when the maximum grows, rescales the running output $\mathbf{o}$ by $e^{m - m'}$ and forgets to rescale $\ell$. Which rows of the result are wrong, and which are right?
2. A kernel marks masked scores with the most negative float and has no test for an all-masked tile. A sequence is left-padded, so the first tile of some rows is entirely padding. Predict the output for a row that has real keys later in the sequence, and for a row of padding only.
3. "Tiled attention is faster because it does fewer FLOPs." Find the error, and say what it does and does not save.
4. At $d_h = 128$ and 48 KiB of fast memory, tiled attention moved 2.89 times the words of the chain in this chapter's table. A colleague concludes that flash attention is useless on such a device. What is right and what is wrong in the conclusion?
5. A decode kernel combines the partial outputs $\mathbf{o}_s/\ell_s$ of the key ranges by averaging them with equal weights. Construct two ranges for which it is wrong, and say what weight each range should carry.
6. A flash backward pass saves $\mathrm{lse}_i$ but computes $D_i$ as $\sum_j p_{ij}\,dp_{ij}$ over the tile it is working on only. In which gradients does the error appear, and does the check against central differences find it at a sequence of one tile?
7. Doubling the query rows per block halves the traffic of the keys and values. What stops a kernel from taking the block as large as it likes?
8. A flash kernel and a materialised chain disagree in the sixth decimal on a float32 model. The test that compares them uses a tolerance of $10^{-4}$ of the largest magnitude. Is that a defect of the kernel?

## Summary

Attention needs a matrix of $T^2$ scores only if the softmax of a row has to be finished before the weighted sum begins. It does not. The online softmax keeps, for each row, the running maximum $m$, the running sum $\ell$ of exponentials relative to it, and the running weighted sum $\mathbf{o}$ of values, and a new chunk of keys updates the triple by multiplying the old sum by $e^{m - m'}$, exactly. The same identity gives a merge of the states of two sets of keys that is associative and has an identity, so keys can be read in any grouping, in tiles by one workgroup or in ranges by many.

Tiled attention walks key and value tiles through shared memory for a block of query rows and never writes a score; the bytes it moves are $2T d_h + 2T d_h\lceil T/B_q \rceil + T$ words, $\Theta(T^2 d_h^2/M)$ for a fast memory of $M$ floats, against $4T^2 + 4T d_h$ for the chain, the same arithmetic plus a small rescale, and a saving of traffic only when the block has more than half as many rows as the head is wide. A kernel run on the simulator, ragged edges included, agreed with materialised attention; the causal mask lets whole tiles be skipped and cuts the diagonal ones, with the hazard of a tile that is masked for a row and a sentinel that is right only because a later visible key erases it. The backward pass saves $T$ numbers, the log-sum-exp, rebuilds each weight as $e^{s - \mathrm{lse}}$, replaces the row sum of the softmax rule by $D_i = d\mathbf{o}_i \cdot \mathbf{o}_i$, and costs five to seven products instead of four; it passed a central-difference check, and three plausible mistakes each failed it where the pattern said they would. For decode, where there is one query, the keys are split across workgroups and the partial states merged, a result checked on lengths that straddle tiles and by the mistake of a missing factor. Everything agrees with the materialised path to tolerance, not to the bit, because the order of the additions differs.

Brain's kernels carry these constraints on a card with no subgroup operations: lanes across the head width to fit the registers, a tile of 8 keys, 64 or 32 query rows per workgroup, an exact portable kernel that is the reference for faster native ones, and a selector that chooses by what the device reports. The next chapter changes the mask and the question. Bidirectional encoders read a whole text at once, so every tile is a full tile, and the matrix that the models of search and classification have to avoid is the same one; what is new there is what the representation is trained to be.

## References

- M. Milakov and N. Gimelshein, "Online normalizer calculation for softmax", 2018.
- M. Rabe and C. Staats, "Self-attention does not need $O(n^2)$ memory", 2021.
- T. Dao, D. Y. Fu, S. Ermon, A. Rudra and C. Re, "FlashAttention: fast and memory-efficient exact attention with IO-awareness", 2022.
- T. Dao, "FlashAttention-2: faster attention with better parallelism and work partitioning", 2023.

## Exercises

1. **[Calculate]** Run the online softmax by hand on the scores $(1, 3, 2, 0)$ in chunks of two, giving $m$ and $\ell$ after each chunk, and the final weights to three decimals. Check them against the stable softmax of the row.
2. **[Calculate]** One layer has $H = 32$ heads, width $D = 4096$ and $T = 100{,}000$ tokens. Give the bytes of its score matrix in float32, the bytes of $\mathbf{Q}$, $\mathbf{K}$, $\mathbf{V}$ and $\mathbf{O}$, and the ratio.
3. **[Calculate]** A workgroup has $M = 8192$ floats of fast memory and $d_h = 32$. Find the largest $B$ with $4B d_h + B^2 \le M$, and the words moved by tiled attention at $T = 4096$ as a fraction of $4T^2 + 4T d_h$.
4. **[Calculate]** For $T = 16$ and blocks of 4 rows against tiles of 4 keys under the causal mask, count the full, diagonal and skipped tiles and the fraction of $T^2$ scores computed.
5. **[Calculate]** One query row has weights $\mathbf{p} = (0.75, 0.25)$ on values $\mathbf{v}_0 = (2, 0)$ and $\mathbf{v}_1 = (0, 4)$, and receives $d\mathbf{o} = (1, 2)$. Compute $c = \sum_j p_j\,dp_j$ directly and $D = d\mathbf{o}\cdot\mathbf{o}$, then $d\mathbf{s}$ by $p_j(dp_j - D)$, and check that its entries sum to zero.
6. **[Calculate]** Merge the states $A = (m, \ell, \mathbf{o}) = (1, 2, (1, 3))$ and $B = (3, 1, (2, 0))$, and give the attention output $\mathbf{o}/\ell$ of the union.
7. **[Implement]** Implement the online softmax and the weighted sum of values from the text and reproduce `flash_attention.online_row` and `flash_attention.online_values`: the state after each chunk of the printed row and the final weights and output, to the printed digits.
8. **[Implement]** Implement the tiled forward kernel of this chapter for the simulator of Volume IV (or a plain loop nest that counts its loads) and reproduce `flash_attention.simulated_forward`: the generator, the draw order, the loads (308 and 252), the stores and the agreement with materialised attention.
9. **[Implement]** Implement the tiled backward pass of the text, with the query-side and key-side passes, and reproduce `flash_attention.backward_check` (agreement with the materialised backward to rounding error and with central differences to about $10^{-8}$), then the three mistakes of `flash_attention.wrong_backward_passes`, whose wrong gradients should differ from the right ones by amounts that agree to one decimal.
10. **[Implement]** Implement the split decode with its combine and reproduce `flash_attention.decode_split`: the draw order, the ranges with keys and the empty ranges for each length, the agreement to rounding error, and the error of the combine that drops the factor to three decimals.
11. **[Investigate]** Reproduce `flash_attention.float32_numerics` in float32 with your own library `exp`, and vary the tile size and the order of the sums. How does the count of bit-identical outputs depend on each? Is there a tile size at which the tiled result is the closest to the double-precision answer, and is that a property of the algorithm or of the inputs?
12. **[Investigate]** Replace the causal mask by a sliding window in which a row sees only the 8 most recent keys, itself included. Classify the tiles of `flash_attention.causal_tiles` for $T = 24$, blocks of 4 and tiles of 4 as skipped, full and cut, count the scores computed against the $8T - 28$ the window allows, and say what the kernel must change to skip the tiles that are now far in the past.

## Key terms

- **online softmax**: the way of computing a softmax and the weighted sum of values in one pass over a row of scores read in chunks, by carrying the running maximum $m$, the running sum $\ell$ of exponentials relative to it and the running weighted sum $\mathbf{o}$, and multiplying the old sum and the old output by $e^{m - m'}$ whenever a chunk raises the maximum from $m$ to $m'$. <!-- concept: online-softmax -->
- **tiled attention**: the schedule that computes attention for a block of query rows by walking tiles of keys and values through fast memory, forming the block of scores of each tile only on chip and folding it into the running state of its rows, so that no score matrix is ever written to global memory. <!-- concept: tiled-attention -->
- **flash attention**: tiled attention with the online softmax in one fused kernel: exact attention whose only results are the output and one log-sum-exp per row, using memory linear in the sequence length and global-memory traffic of $\Theta(T^2 d_h^2 / M)$ for a fast memory of $M$ numbers. <!-- concept: flash-attention -->
- **flash attention backward**: the backward pass of attention that saves only the output and the row log-sum-exp, rebuilds each weight as $e^{s - \mathrm{lse}}$ tile by tile, replaces the row sum of the softmax rule by $D_i = d\mathbf{o}_i \cdot \mathbf{o}_i$, and produces the gradients of the queries, keys and values without storing any $T \times T$ matrix. <!-- concept: flash-attention-backward -->

## Answers

**Recall questions.**

1. $c_i = \sum_j p_{ij}\,dp_{ij}$, the weights of the row dotted with their gradients. It enters $dS$, so it affects $d\mathbf{Q}$ and $d\mathbf{K}$, which are computed from $d\mathbf{S}$, and not $d\mathbf{V} = \mathbf{P}^\top d\mathbf{O}$.
2. The largest entry is subtracted first, so the softmax of $(1000, 1001)$ is that of $(-1, 0)$, $(0.269, 0.731)$. It costs one extra pass over the vector, to find the maximum, before the sum of exponentials and the division.
3. $\mathrm{lse}(0, \ln 3) = \ln(1 + 3) = \ln 4 = 1.386$. The sum is kept from overflowing by factoring out the largest entry: $\mathrm{lse}(\mathbf{z}) = m + \ln \sum_i e^{z_i - m}$ with $m = \max_i z_i$, so every exponent is at most 0.
4. $2n^3/T$ numbers, in the square case: each number loaded into shared memory is used by $T$ products before it is replaced, so a tile of $T \times T$ numbers serves $T^3$ multiply-adds, where one invocation per output would load each number for every product.
5. The first barrier makes the tile complete before any invocation reads it, so no invocation reads a number another has not yet written. The second stops a fast invocation from beginning the next step and overwriting the tile while a slow one is still reading it.
6. $(1 + 2^{-24}) + 2^{-24} = 1$, since each addition rounds back to 1, and $1 + (2^{-24} + 2^{-24}) = 1 + 2^{-23} = 1.00000012$. Addition of floats is not associative, so a tree of additions and a loop give different last bits for the same numbers, and neither is wrong.

**Check your understanding.**

1. The first chunk's rows are right (the correction multiplies zero there), and so are rows whose maximum never grows after the first chunk. A row whose maximum grows later has an $\ell$ that is too large by the factor $e^{m' - m}$ it should have been shrunk by: its output $\mathbf{o}/\ell$ comes out too small, every component by the same factor, and the output is a correctly directed vector of the wrong length. The error is silent in a test whose row maximum always sits in the first chunk.
2. For a row with real keys later, the output is correct: the masked first tile contributes garbage to $\ell$ and $\mathbf{o}$ at weight $e^0 = 1$, and the first tile with a real score raises the maximum to a finite value and multiplies the garbage by $e^{-3.4 \times 10^{38} - m'} = 0$. For a row of padding only, no tile ever raises the maximum, the garbage stays, and the output is the plain mean of the values of the row, neither zero nor an error.
3. The FLOPs are the same: $4T^2 d_h$ for the products, plus a small rescale (the table shows the total within 3 percent of the chain at tiles of 8 keys). What tiling saves is the memory of the matrix, and, when the block holds more than $d_h/2$ rows, bytes moved. Whether that makes a kernel faster depends on whether it was limited by those bytes.
4. Right: at that size of fast memory the block is 22 rows, fewer than $d_h/2 = 64$, so the traffic is higher than the chain's lower bound. Wrong: the matrix is still not stored, so the sequence length the device can hold is no longer limited by $T^2$, and a larger $M$ (registers count, as the kernels of brain use them for the queries and outputs) raises $B$. The conclusion confuses a saving of capacity with a saving of traffic.
5. Take two ranges with the same $\mathbf{o}_s/\ell_s$ but different totals: range 0 has one key with score 0 and value $(1, 0)$, range 1 has three keys with score 0 and value $(0, 1)$. The partial outputs are $(1, 0)$ and $(0, 1)$, the equal-weight average is $(0.5, 0.5)$, and the attention over the four keys is $(0.25, 0.75)$. Each range should carry the weight $\ell_s e^{m_s - m} / \ell$, its share of the total normalizer: here $1/4$ and $3/4$.
6. $D_i$ over a tile is a partial sum, so $dS_{ij}$ is wrong in every row whenever there is more than one tile: $d\mathbf{Q}$ and $d\mathbf{K}$ are wrong, $d\mathbf{V}$ is right. At a sequence of one tile the partial sum is the whole sum and the error is invisible, which is one more reason to check a gradient with sizes that no tile divides.
7. The fast memory that holds the block: the query rows and the output rows (and, in registers, the states) grow with $B_q$, along with the scores of a tile, so the footprint $4B d_h + B^2$ must stay within $M$, and on a GPU the registers of an invocation limit how much a thread can keep. A very large block also leaves fewer workgroups for the device to run.
8. No. A difference of a few units of the float32 roundoff is what two correct computations with different addition orders look like (the table of the section on numerics). The tolerance is stated against the largest magnitude so that it is neither tighter than the arithmetic allows nor loose enough to hide a wrong term; a defect would show as a difference of order 1 relative, as in the table of wrong backward passes.

**Exercises.**

1. After chunk 0 (scores 1, 3): $m = 3$, $\ell = e^{-2} + 1 = 1.1353$. After chunk 1 (scores 2, 0): $m = 3$ (unchanged, correction 1), $\ell = 1.1353 + e^{-1} + e^{-3} = 1.1353 + 0.3679 + 0.0498 = 1.5530$. Weights $e^{s - 3}/1.5530$: $(0.087, 0.644, 0.237, 0.032)$, the softmax of the row.
2. Score matrix $4 \cdot 32 \cdot 10^{10} = 1.28 \times 10^{12}$ bytes (1.28 TB); the four matrices $4 \cdot 10^5 \cdot 4096 \cdot 4 = 6{,}553{,}600{,}000$ bytes; ratio $T/(4 d_h)$ with $d_h = 128$, which is $195.3$.
3. $B = \lfloor\sqrt{4d_h^2 + M} - 2 d_h\rfloor = \lfloor\sqrt{12288} - 64\rfloor = 46$, footprint $4 \cdot 46 \cdot 32 + 46^2 = 8004 \le 8192$. Words tiled over words of the chain at $T = 4096$: $0.353$.
4. 16 tiles in a $4 \times 4$ grid: 6 full (below the diagonal), 4 diagonal, 6 skipped; scores computed $(6 + 4) \cdot 16 = 160$ of 256, a fraction of 0.625 (against $17/32 = 0.531$ allowed).
5. $\mathbf{o} = 0.75(2, 0) + 0.25(0, 4) = (1.5, 1)$. $dp_0 = d\mathbf{o}\cdot\mathbf{v}_0 = 2$, $dp_1 = 8$, so $c = 0.75 \cdot 2 + 0.25 \cdot 8 = 3.5$, and $D = 1 \cdot 1.5 + 2 \cdot 1 = 3.5$, equal. $d\mathbf{s} = (0.75(2 - 3.5), 0.25(8 - 3.5)) = (-1.125, 1.125)$, which sums to zero as a softmax gradient must.
6. $m = 3$, $\ell = 2e^{-2} + 1 = 1.2707$, $\mathbf{o} = (1 \cdot e^{-2} + 2,\; 3e^{-2} + 0) = (2.1353, 0.4060)$. The output is $\mathbf{o}/\ell = (1.6805, 0.3195)$.
7. The state after each chunk of the printed row and the final weights are in the worked blocks; an implementation that adds in another order should agree to the printed six decimals.
8. 308 loads without the mask ($44$ for $\mathbf{Q}$ plus $3 \cdot 88$ for the keys and values, each workgroup loading all 11 rows), 252 with it, 55 stores ($44$ outputs and $11$ log-sum-exps), and agreement with materialised attention to rounding error.
9. The correct tiled gradients agree with the materialised ones to rounding error and with central differences to a relative $10^{-8}$ or $10^{-9}$; the wrong ones differ by about 1 in the tensors the table names, and $d\mathbf{V}$ is right when only $D_i$ is dropped.
10. For each length, the number of ranges holding keys is $\lceil \lceil L/8 \rceil / 2 \rceil$ (1, 1, 1, 1, 3 and 7) and the rest of the 7 are empty; the combine agrees with attention to rounding error, and the combine without the factor is right while one range holds keys and wrong by 0.247 and 0.100 at lengths 41 and 100.
11. An investigation. Expect the count of identical outputs to move with the tile size and with the order of the sums (it is a function of where roundings fall) while the largest difference from the double-precision answer stays at a few units of $2^{-24}$ for every version; the chapter's table reports one set of inputs, and a different set will shift the counts.
12. An investigation. With a window of 8 keys (row $i$ sees keys $i - 7, \dots, i$) and blocks and tiles of 4, the 36 tiles of $T = 24$ are 21 skipped, 5 full and 10 cut, and the kernel computes $10 \cdot 16 + 5 \cdot 16 = 240$ scores against the 164 the window allows. A tile is skipped when it lies wholly after the last row of the block or wholly before the window of the first row, so the computed tiles form a band two or three tiles wide and the work is linear in $T$. The kernel must therefore also start its loop at the first tile that can meet the window of the block's first row, as the causal key-side pass already starts at the first query block that can see the key, and not at tile 0.
