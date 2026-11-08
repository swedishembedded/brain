<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Making Matrix Multiplication Fast

*How does a matrix multiply go from a three-line loop to running near the hardware's peak?*

Multiply two 4 by 4 matrices with the definition of Volume I, Chapter 4: output $C_{ij}$ is the dot product of row $i$ of $A$ and column $j$ of
$B$. As three nested loops it is this, with the sum kept in a variable $\mathrm{acc}$:

```text
for i in 0 .. M:
    for j in 0 .. N:
        acc = 0.0
        for k in 0 .. K:
            acc = acc + A[i, k] * B[k, j]
        C[i, j] = acc
```

The next block runs it on two small integer matrices (integers, so that every sum is exact) and counts what it does:

<!-- worked: matmul_fast.naive_small -->
A[i, k] = (7 i + 3 k) mod 11 - 5 and B[k, j] = (5 k + 2 j) mod 13 - 6, both [4, 4], integers so that every sum is exact. The loop is `for i, for j, acc = 0, for k: acc = acc + A[i, k] B[k, j]`, then `C[i, j] = acc`.

A:

| A | col 0 | col 1 | col 2 | col 3 |
|---|---:|---:|---:|---:|
| row 0 | -5 | -2 | 1 | 4 |
| row 1 | 2 | 5 | -3 | 0 |
| row 2 | -2 | 1 | 4 | -4 |
| row 3 | 5 | -3 | 0 | 3 |

B:

| B | col 0 | col 1 | col 2 | col 3 |
|---|---:|---:|---:|---:|
| row 0 | -6 | -4 | -2 | 0 |
| row 1 | -1 | 1 | 3 | 5 |
| row 2 | 4 | 6 | -5 | -3 |
| row 3 | -4 | -2 | 0 | 2 |

C = A B:

| C | col 0 | col 1 | col 2 | col 3 |
|---|---:|---:|---:|---:|
| row 0 | 20 | 16 | -1 | -5 |
| row 1 | -29 | -21 | 26 | 34 |
| row 2 | 43 | 41 | -13 | -15 |
| row 3 | -39 | -29 | -19 | -9 |

One output: C[1, 2] = (2)(-2) + (5)(3) + (-3)(-5) + (0)(0) = 26.

| count | value |
|---|---:|
| multiply-adds m n k | 64 |
| FLOPs 2 m n k | 128 |
| loads, 2 k per output | 128 |
| distinct numbers in A and B | 32 |
| loads per distinct number | 4 |
| stores | 16 |
<!-- /worked -->

Every output costs $K = 4$ multiplications and 4 additions, and 8 loads, 4 from a row of $A$ and 4 from a column of $B$. There are 16 outputs, so
the loop performs $2 m n k = 128$ floating-point operations and 128 loads, yet the two matrices hold only 32 numbers: each number was loaded four times. The
loop has no memory. It forgets the row of $A$ the moment an output is finished, then loads it again for the next output.

At this size nobody minds. At $n = 1024$, a size a single layer of a modest network reaches, the same loop performs $2 n^3 = 2{,}147{,}483{,}648$ operations, and its
loads are as many numbers as operations. Every layer of Volume III's networks is such a product, and a training step runs one for each layer, forwards and backwards; Chapter 3 gave the vocabulary to say what
that costs: the arithmetic intensity of this loop is about a quarter of an operation per byte, far to the left of the ridge point of any machine of that chapter, so a machine
that could do 40,000 (GFLOP/s) is fed by a memory that allows 200. Yet the product itself, counted as Chapter 3 counted it, reads each input once and has an intensity of
$n/6 = 170.7$, to the right of both ridge points. The gap between 0.25 and 170.7 belongs to the loop and not to the product, and this chapter closes it, in
four steps: reuse in the cache of a CPU, reuse in the shared memory of a GPU workgroup, reuse in the registers of one invocation, and finally arithmetic units built for the
product itself. A last section asks what these rearrangements do to the numbers, since a sum added in a different order is, in floating point, a different sum.

Five ideas from earlier chapters carry the argument, from the cost of a product (Volume IV, Chapter 3) to the order of floating-point additions (Volume I, Chapter 15). Try them from memory first; the answers are at the end of the chapter, under "Recall questions".

1. (Volume IV, Chapter 3, arithmetic intensity.) How many floating-point operations does $C = AB$ take for two $[512, 512]$ fp32 matrices, how many bytes must cross
   the memory boundary at the very least (read $A$ and $B$ once, write $C$ once), and what is the intensity?
2. (Volume IV, Chapter 2, cache lines.) A $[256, 256]$ fp32 matrix is stored row-major and read down one column. How many different 64-byte cache lines does the column touch, and
   how many numbers of each line are used by that column?
3. (Volume IV, Chapter 4, shared memory.) Why does a workgroup that stages a tile in shared memory need a barrier between the invocations that write the tile and the
   invocations that read it?
4. (Volume IV, Chapter 4, occupancy.) A kernel uses 40 registers per invocation, 256 invocations per workgroup and 2,048 bytes of shared memory per workgroup. How many
   workgroups are resident on one compute unit of the model GPU (65,536 registers, 65,536 bytes of shared memory, 2,048 resident invocations, at most 16 workgroups)?
5. (Volume I, Chapter 15, rounding.) In fp32, with unit roundoff $u = 2^{-24}$, is $(1 + 2^{-24}) + 2^{-24}$ equal to $1 + (2^{-24} + 2^{-24})$? Why does it matter to
   a program that adds the same numbers in a different order?

## Counting the three-line loop

Before changing the loop we say exactly what it costs in the two units of Chapter 3, operations and bytes, and where it stands on the roofline. The loop above is **naive matmul**: the product computed exactly as its definition reads, one output at a time, each output a dot product of a row of $A$ and a column of $B$,
with nothing kept from one output to the next. On a GPU the same idea is one invocation per output (Chapter 5):

```text
kernel matmul_naive(A: array<f32>[M, K], B: array<f32>[K, N], C: inout array<f32>[M, N], M: int, N: int, K: int) workgroup n_wg:
    if g >= M * N: return
    i = g // N
    j = g mod N
    acc = 0.0
    for k in 0 .. K:
        acc = acc + A[i, k] * B[k, j]
    C[i, j] = acc

launch matmul_naive(A, B, C, M, N, K) over ceil(M * N / n_wg) workgroups
```

Adjacent invocations have adjacent $j$, so the loads $B[k, j]$ of one subgroup are consecutive numbers of a row (Chapter 4's coalescing) and the loads $A[i, k]$ are one number that every
lane shares. Each memory instruction is well formed. The waste is elsewhere: every invocation fetches its own copy of a whole row and a whole column from device memory. Count it for
square matrices of side $n$. The loop performs $n^2$ outputs times $n$ multiply-adds, $N_{\mathrm{flop}} = 2n^3$. It loads $2n$ numbers per output, $2n^3$ in all, and stores $n^2$,
so with 4-byte numbers

$$
N_{\mathrm{byte}} = 4\,(2n^3 + n^2), \qquad \mathcal{I} = \frac{2n^3}{4(2n^3 + n^2)} = \frac{1}{4 + 2/n}.
$$

<!-- worked: matmul_fast.naive_counts -->
C = A B, all sides n, fp32, one invocation per output with a K-loop: loads 2 n^3, stores n^2, bytes 4 (2 n^3 + n^2) if every load goes to main memory; FLOPs 2 n^3. The lower bound reads A and B once and writes C once: 12 n^2 bytes. GPU of Chapters 3 and 4: F_pk = 40,000 (GFLOP/s), BW = 800 (GB/s), attainable = min(F_pk, intensity x BW).

| n | FLOPs | loads | bytes | FLOP per byte | bytes, lower bound | FLOP per byte, bound | GPU attainable (GFLOP/s) | fraction of peak |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 128 | 128 | 576 | 0.2222 | 192 | 0.67 | 177.8 | 0.0044 |
| 64 | 524,288 | 524,288 | 2,113,536 | 0.2481 | 49,152 | 10.67 | 198.4 | 0.0050 |
| 256 | 33,554,432 | 33,554,432 | 134,479,872 | 0.2495 | 786,432 | 42.67 | 199.6 | 0.0050 |
| 1,024 | 2,147,483,648 | 2,147,483,648 | 8,594,128,896 | 0.2499 | 12,582,912 | 170.67 | 199.9 | 0.0050 |
| 4,096 | 137,438,953,472 | 137,438,953,472 | 549,822,922,752 | 0.2500 | 201,326,592 | 682.67 | 200.0 | 0.0050 |
<!-- /worked -->

The intensity approaches $1/4$ from below and is already 0.2499 at $n = 1024$: the intensity of a dot product (Chapter 3), however large the matrix. On the GPU of Chapters 3 and
4 (peak 40,000 (GFLOP/s), bandwidth 800 (GB/s), ridge point 50) the attainable rate is $\min(40000, 0.25 \times 800) = 200$ (GFLOP/s), a fraction 0.005 of the peak; on the CPU
(peak 800, bandwidth 80, ridge 10) it is $0.25 \times 80 = 20$, a fraction 0.025. The last two columns of the table are the other half of the story. The lower bound on bytes, $12 n^2$, gives
an intensity of $n/6$ that grows with the matrix: 170.67 at $n = 1024$, above both ridge points. So the product can be compute-bound on either machine, and the naive loop moves
about 683 times as many bytes as it needs to at $n = 1024$ ($8{,}594{,}128{,}896 / 12{,}582{,}912$).

Where do the extra bytes come from? An element $A_{ik}$ is used by every output of row $i$, $n$ of them, and the loop loads it $n$ times. The product has $2n^3$ operations on only $2n^2$
distinct inputs, so every input can be used $n$ times, and the whole art of a fast matrix multiplication is to make those $n$ uses happen while the number is still in a memory close to the arithmetic. The first
place to see it is the cache of a CPU, where the machinery is Chapter 2's.

## Loop order and the cache

The same reuse problem appears first on a CPU, where the machinery is Chapter 2's, so a reordered or blocked loop can be priced in cache lines before the GPU adds a memory of its own. Run the loop on the cache simulator of Chapter 2: a 32 KiB cache of 64-byte lines with least-recently-used replacement, either 8-way (64 sets) or fully associative. The three matrices are
fp32, row-major, and each starts on a line boundary. Three loop nests are compared. **ijk** is the loop above: the inner loop reads $A[i, k]$ along a row and $B[k, j]$ down a column. **ikj**
swaps the inner two loops, so that the inner loop runs along row $k$ of $B$ and row $i$ of $C$ and $A[i, k]$ is a constant for it. The **blocked** nest cuts all three loops into blocks of
$16$ and runs ijk inside each block, so that a $16 \times 16$ block of each matrix is used for a while before the next. All three perform the same $2n^3$ operations. Only the lines fetched differ:

<!-- worked: matmul_fast.loop_order_misses -->
C = A B, n by n fp32 row-major matrices at addresses 0, 4 n^2 and 8 n^2 (so each starts on a line boundary), the cache of Chapter 2: 32 KiB, 64-byte lines, least recently used, 8-way (64 sets) or fully associative (512 ways). Counted accesses: ijk reads A[i, k] and B[k, j] in the inner loop and touches C[i, j] once per (i, j); ikj reads A[i, k] once per (i, k) and then B[k, j] and C[i, j] in the inner loop; the blocked nest runs ijk inside blocks of 16 x 16 x 16 with the block loops in order i, j, k and touches C[i, j] once per block. 'Intensity' = 2 n^3 / (64 x line misses): FLOPs per byte of lines fetched. Compulsory = 3 n^2 x 4 / 64 lines, the least any order can fetch.

| n | loop order | compulsory lines | line misses, 8-way | line misses, full | FLOP per byte, 8-way | FLOP per byte, full |
|---:|---|---:|---:|---:|---:|---:|
| 64 | i, j, k (the dot-product order) | 768 | 768 | 768 | 10.667 | 10.667 |
| 64 | i, k, j (row of B at a time) | 768 | 768 | 768 | 10.667 | 10.667 |
| 64 | ijk inside 16 x 16 x 16 blocks | 768 | 768 | 768 | 10.667 | 10.667 |
| 96 | i, j, k (the dot-product order) | 1,728 | 56,448 | 56,448 | 0.490 | 0.490 |
| 96 | i, k, j (row of B at a time) | 1,728 | 56,448 | 56,448 | 0.490 | 0.490 |
| 96 | ijk inside 16 x 16 x 16 blocks | 1,728 | 4,608 | 4,608 | 6.000 | 6.000 |
| 128 | i, j, k (the dot-product order) | 3,072 | 2,130,960 | 133,120 | 0.031 | 0.492 |
| 128 | i, k, j (row of B at a time) | 3,072 | 133,120 | 133,120 | 0.492 | 0.492 |
| 128 | ijk inside 16 x 16 x 16 blocks | 3,072 | 11,768 | 10,240 | 5.569 | 6.400 |
<!-- /worked -->

Read the table by size. At $n = 64$ the matrix $B$ is 16 KiB: it, a row of $A$ and a row of $C$ stay in the cache, and every order fetches the compulsory 768 lines, each number once, the intensity $n/6$ of Chapter 3. At
$n = 96$ the matrix $B$ is 36 KiB and no longer fits. Both ijk and ikj then stream the whole of $B$ once for every row $i$ of $C$: $n \cdot n^2/16 = 55{,}296$ lines plus the 1,152 of $A$ and $C$, the 56,448 of the table. Every line of $B$ is fetched once per row of $C$ and all 16 of its numbers are used, yet the intensity is $2n^3 / (64 \cdot n^3/16) = 1/2$ of an operation per byte: each number of $B$ is brought in $n$ times and used once each time. At $n = 128$ the 8-way cache adds the effect
of Chapter 2: a row of $B$ is 512 bytes, the lines of a column fall in only 8 of the 64 sets, and ijk misses on almost every access, 2,130,960 lines against the 133,120 of ikj, an intensity of $1/32$. Loop order fixed that
one pathology and nothing else: ikj still streams $B$ once per row of $C$.

The blocked nest, the cache blocking that Goto and van de Geijn built the fast matrix products of CPU libraries around, changes the count itself. With blocks of $T = 16$ the whole of $B$ is streamed once per block-row of $C$, $n/T$ times instead of $n$: at $n = 128$ that is $8 \times 1{,}024 = 8{,}192$ lines for $B$ plus 1,024 for $A$
and 1,024 for $C$, the 10,240 of the fully associative column. The traffic fell by a factor $T$, which is the law of every section that follows: **a number used $T$ times from a near memory costs $1/T$ of the traffic of a number loaded for every
use.** On this model CPU the blocked nest reaches 6.4 operations per byte, below the ridge point of 10, and a larger block would be needed to cross it (the model of Chapter 3 gave 15.88 for $T = 64$).

The GPU has the same structure, with two differences. Its near memory is not decided by an eviction rule but by the program, in the shared memory of Chapter 4. And it has thousands of invocations to organize, so the block of the
output, the *tile*, is also the unit of work of a workgroup.

## Tiling in shared memory

A workgroup that computes a tile of $C$ and keeps the matching tiles of $A$ and $B$ in shared memory uses every number it loads from device memory $T$ times. We count the loads on the simulator of Chapter 5 and derive their formula.

**Tiled matmul** is the product computed one tile of the output at a time: a workgroup of $T \times T$ invocations owns a $T \times T$ tile of $C$, one output per invocation, and walks along the shared dimension in steps of $T$. In each step the workgroup copies a
$T \times T$ tile of $A$ and a $T \times T$ tile of $B$ from device memory into shared memory, each invocation copying one number of each, waits at a barrier, and then every invocation does $T$ multiply-adds from shared memory. The figure
draws one step for $n = 16$ and $T = 4$.

![A 16 by 16 product drawn as 4 by 4 grids of 4 by 4 blocks of A, B and C. One workgroup owns block (3, 1) of C; the picture shows its last step, where block (3, 3) of A and block (1, 1) of B pass through the shared memory box and the workgroup does 4 multiply-adds per output. The text under it counts 128 loads against 512 for one invocation per output.](img/tiled-matmul.svg)

In the book's pseudocode, with the guards that keep ragged edges correct (the barriers are outside every `if`, the rule of Chapter 5):

```text
kernel matmul_tiled(A: array<f32>[M, K], B: array<f32>[K, N], C: inout array<f32>[M, N], M: int, N: int, K: int, T: int) workgroup T * T:
    shared a_tile: array<f32>[T * T]
    shared b_tile: array<f32>[T * T]
    ly = l // T
    lx = l mod T
    row = w_y * T + ly                       # the output row of this invocation
    col = w_x * T + lx                       # the output column
    acc = 0.0
    for t in 0 .. ceil(K / T):
        ka = t * T + lx                      # the column of A this invocation copies
        kb = t * T + ly                      # the row of B this invocation copies
        a_tile[ly * T + lx] = 0.0
        if row < M and ka < K:
            a_tile[ly * T + lx] = A[row, ka]
        b_tile[ly * T + lx] = 0.0
        if kb < K and col < N:
            b_tile[ly * T + lx] = B[kb, col]
        barrier                              # the tiles are complete
        for j in 0 .. T:
            acc = acc + a_tile[ly * T + j] * b_tile[j * T + lx]
        barrier                              # everyone is done reading before the next step overwrites
    if row < M and col < N:
        C[row, col] = acc

launch matmul_tiled(A, B, C, M, N, K, T) over ceil(N / T) by ceil(M / T) workgroups
```

There are two barriers per step, and each has its own job. The first makes the tile complete before anyone reads it. The second stops a fast invocation from starting the next step and overwriting a tile that a slow invocation is still reading;
dropping it passes every test with a single step ($K \le T$) and fails with more, the same signature as the missing barrier of Chapter 5. Elements outside the matrices are loaded as zeros, so the inner loop needs no guard and
a partial last step adds zero terms.

Run it. The simulator of Chapter 5 executes the body of every invocation of every workgroup; the kernel counts each load it issues, and the product is compared with the direct one. The first product is $16 \times 16$; the second has
ragged sides, so that tiles at the edges are partial:

<!-- worked: matmul_fast.tiled_trace -->
Integer matrices A[i, k] = (7 i + 3 k) mod 11 - 5 and B[k, j] = (5 k + 2 j) mod 13 - 6 (as in `matmul_fast.naive_small`), run on the kernel simulator of Chapter 5 (32-lane subgroups, 32-byte blocks). Loads are counted by the kernel as it executes them; an invocation that falls outside the matrices loads nothing. Formula: naive 2 m n k; tiled k (m ceil(n / T) + n ceil(m / T)). 'Load blocks' are the simulator's 32-byte transactions for loads, counted as follows. A memory instruction is one load statement of the kernel at one step of its loops: for the naive kernel the load of A and the load of B at one k; for the tiled kernel the load of A and the load of B at one step e of the cooperative loop in one k-chunk of one workgroup. The invocations of a subgroup (32 consecutive invocations of a workgroup; a workgroup of 16 is one partial subgroup) that reach an instruction issue it together, and one that has returned or whose element lies outside the matrix does not issue it, so a skipped load never moves the later loads of that invocation into another instruction. An instruction costs one transaction per distinct 32-byte block (8 consecutive words, aligned to a multiple of 8 words) that its addresses touch. A, B and C sit in one flat memory of 4-byte words, in that order, each buffer starting at the first multiple of 8 words at or after the end of the previous one: for [6, 7] by [7, 10], A is words 0 to 41, B words 48 to 117 and C words 120 to 179 (for [16, 16] by [16, 16] all three are multiples of 8 anyway).

| product | kernel | global loads, counted | global loads, formula | load blocks | equals direct |
|---|---|---:|---:|---:|---|
| [16, 16] by [16, 16] | naive, 16 invocations per workgroup | 8,192 | 8,192 | 768 | yes |
| [16, 16] by [16, 16] | tiled, T = 4 (16 invocations) | 2,048 | 2,048 | 512 | yes |
| [16, 16] by [16, 16] | tiled, T = 8 (64 invocations) | 1,024 | 1,024 | 128 | yes |
| [6, 7] by [7, 10] | naive, 16 invocations per workgroup | 840 | 840 | 114 | yes |
| [6, 7] by [7, 10] | tiled, T = 4 (16 invocations) | 266 | 266 | 84 | yes |
<!-- /worked -->

The naive kernel loads $2mnk = 8{,}192$ numbers for the $16 \times 16$ product, the tiled kernel with $T = 4$ loads 2,048 and with $T = 8$ loads 1,024: the count is divided by $T$. The formula behind the column is exact for any shape. A number
$A_{ik}$ is copied by every workgroup of the block-row that contains $i$, that is once for each of the $\lceil n/T \rceil$ tile columns of $C$, and a number $B_{kj}$ once for each of the $\lceil m/T \rceil$ tile rows, so

$$
\text{global loads} = k \,\bigl( m \lceil n/T \rceil + n \lceil m/T \rceil \bigr) \;\longrightarrow\; \frac{2 n^3}{T} \quad (m = n = k, \; T \mid n).
$$

The ragged case agrees: $7\,(6 \cdot 3 + 10 \cdot 2) = 266$ against the naive $2 \cdot 6 \cdot 10 \cdot 7 = 840$. Chapter 4's count of $2n^3/T$ is the square case, and the counted loads in the table are the simulator agreeing with the formula, not
the formula repeated. (The last column, the 32-byte transactions that the loads cost, falls less than the loads do at this size, 768 to 512 at $T = 4$, because the naive kernel's loads were already well formed, as noted above; it is the count of numbers loaded that carries the reuse.)

Intensity follows from the same count. Add the $n^2$ stores of $C$ and the bytes are $4(2n^3/T + n^2)$, so

$$
\mathcal{I}(T) = \frac{2n^3}{4\,(2n^3/T + n^2)} = \frac{1}{4/T + 2/n} \approx \frac{T}{4},
$$

the formula of Chapter 3's cache tiling, now for a tile in shared memory. The depth of a step, $T$ here, does not appear: a step of depth $b_k$ loads $2Tb_k$ numbers and does $T^2 b_k$ multiply-adds, so $b_k$ cancels. Deepening the
steps changes the number of barriers and the latency exposed, never the traffic. (Commit 455f8054a lists exactly this as a hypothesis it killed with the arithmetic: the intensity of the register-tiled kernel is independent of its step depth.)

How large can $T$ be? Three limits meet. A workgroup has $T^2$ invocations, one per output, and the portable limit of Chapter 4 is 256, so $T \le 16$. The shared memory is $2T^2$ numbers, $8T^2$ bytes, against a limit of 16 KiB. And the
tile competes for the compute unit with other workgroups, so occupancy (Chapter 4) decides how many run at once.

<!-- worked: matmul_fast.tile_size_table -->
C = A B, n = 1024, fp32, workgroup of T x T invocations computing a T x T tile with T x T tiles of A and B in shared memory (2 T^2 numbers, 8 T^2 bytes); global loads 2 n^3 / T; bytes 4 (loads + n^2); 40 registers per invocation as in Chapter 4's table; occupancy by the model of Chapter 4 (65,536 registers and 65,536 bytes of shared memory per compute unit, 2,048 resident invocations, 16 resident workgroups). Portable: at most 256 invocations and 16 KiB of shared memory. Attainable = min(40,000, intensity x 800), in (GFLOP/s).

| T | invocations | shared bytes | portable | resident workgroups | occupancy (%) | limited by | global loads | FLOP per byte | attainable (GFLOP/s) | fraction of peak |
|---:|---:|---:|---|---:|---:|---|---:|---:|---:|---:|
| 2 | 4 | 32 | yes | 16 | 3.1 | workgroup cap | 1,073,741,824 | 0.50 | 400 | 0.010 |
| 4 | 16 | 128 | yes | 16 | 12.5 | workgroup cap | 536,870,912 | 1.00 | 798 | 0.020 |
| 8 | 64 | 512 | yes | 16 | 50.0 | workgroup cap | 268,435,456 | 1.99 | 1,594 | 0.040 |
| 16 | 256 | 2,048 | yes | 6 | 75.0 | registers | 134,217,728 | 3.97 | 3,175 | 0.079 |
| 32 | 1,024 | 8,192 | no | 1 | 50.0 | registers | 67,108,864 | 7.88 | 6,302 | 0.158 |
<!-- /worked -->

$T = 16$ is the largest portable tile, with 256 invocations and 2,048 bytes of shared memory, and the occupancy model leaves it at 6 resident workgroups, 75 percent of the compute unit's slots, limited by registers; $T = 32$ would need 1,024 invocations
and is not portable. Even if it were, its intensity of 7.88 is below the ridge point of 50: the best tile that one-output-per-invocation can offer leaves the product bandwidth-bound, at a fraction 0.079 of the peak for $T = 16$ and 0.158 for $T = 32$. The figure shows why the portable limit, not the memory,
is the wall: the loads fall as $1/T$, and $T$ is stuck at 16 by the number of invocations.

![Global loads of a 1024 by 1024 product on a log-log plot against the side T of the workgroup tile: a straight line of slope minus one from 2 n^3 at T = 1 to 2 n^2 at T = n. Green points for T from 1 to 16 stop at a dashed vertical line at 256 invocations; red points for T = 32, 64 and 128 mark register blocking with r = 2, 4 and 8.](img/loads-vs-tile.svg)

The way past the wall is to stop tying the tile's side to the number of invocations: let one invocation compute more than one output.

## Register blocking

Letting one invocation compute more than one output makes the tile of the output bigger than the workgroup, at a price in registers that we can count. We derive what the block saves, what it costs and where the limit is.

**Register blocking** is giving each invocation an $r \times r$ block of outputs, held in registers, instead of one output. A workgroup of $q \times q$ invocations then computes a tile of side $T = q\,r$ with $q^2$ invocations. For one step along the shared dimension an
invocation reads the $r$ numbers of its rows of $A$ and the $r$ numbers of its columns of $B$ from shared memory, once, into registers, and uses each of the $2r$ numbers $r$ times: $r^2$ multiply-adds from $2r$ shared-memory reads. The pseudocode is the tiled kernel with these changes, a
step depth $b_k$ that is a free parameter, and a loop that gathers the tile cooperatively because the tile has more elements than the workgroup has invocations:

```text
kernel matmul_blocked(A: array<f32>[M, K], B: array<f32>[K, N], C: inout array<f32>[M, N], M: int, N: int, K: int, q: int, r: int, bk: int) workgroup q * q:
    T = q * r                                # side of the workgroup's output tile
    shared a_tile: array<f32>[T * bk]        # a_tile[row * bk + kk]: T rows of A, bk columns
    shared b_tile: array<f32>[bk * T]        # b_tile[kk * T + col]: bk rows of B, T columns
    ty = l // q
    tx = l mod q
    acc = zeros(f32, [r, r])                 # the registers of this invocation
    a_reg = zeros(f32, [r])
    b_reg = zeros(f32, [r])
    for c in 0 .. ceil(K / bk):
        for e in 0 .. ceil(T * bk / (q * q)):
            idx = l + e * q * q
            if idx < T * bk:
                a_tile[idx] = 0.0
                if w_y * T + idx // bk < M and c * bk + idx mod bk < K:
                    a_tile[idx] = A[w_y * T + idx // bk, c * bk + idx mod bk]
                b_tile[idx] = 0.0
                if c * bk + idx // T < K and w_x * T + idx mod T < N:
                    b_tile[idx] = B[c * bk + idx // T, w_x * T + idx mod T]
        barrier
        for kk in 0 .. bk:
            for i in 0 .. r:
                a_reg[i] = a_tile[(ty * r + i) * bk + kk]
            for j in 0 .. r:
                b_reg[j] = b_tile[kk * T + tx * r + j]
            for i in 0 .. r:
                for j in 0 .. r:
                    acc[i, j] = acc[i, j] + a_reg[i] * b_reg[j]
        barrier
    for i in 0 .. r:
        for j in 0 .. r:
            if w_y * T + ty * r + i < M and w_x * T + tx * r + j < N:
                C[w_y * T + ty * r + i, w_x * T + tx * r + j] = acc[i, j]

launch matmul_blocked(A, B, C, M, N, K, q, r, bk) over ceil(N / (q * r)) by ceil(M / (q * r)) workgroups
```

With $r = 1$ and $b_k = T$ this is the tiled kernel of the previous section. Run on the simulator with the same $T = 4$ in three shapes, 16, 4 and 1 invocations per workgroup, on the $8 \times 8$ integer product:

<!-- worked: matmul_fast.blocked_trace -->
The product of two [8, 8] integer matrices (as in `matmul_fast.naive_small`) on the simulator, workgroup tile T = q r = 4 for the last three rows, k-chunk 2. Every count is taken from the running kernel: global loads of A and B, shared-memory reads, fused multiply-adds (each is 1 of the 8^3 = 512 that the product needs). The global loads depend on T only; the shared reads fall as 1 / r.

| kernel | workgroup tile T | invocations per workgroup | outputs per invocation r^2 | global loads | shared reads | FMAs | equals direct |
|---|---:|---:|---:|---:|---:|---:|---|
| naive (one output per invocation) | - | 16 | 1 | 1,024 | 0 | 512 | yes |
| tiled: q = 4, r = 1 | 4 | 16 | 1 | 256 | 1,024 | 512 | yes |
| register-blocked: q = 2, r = 2 | 4 | 4 | 4 | 256 | 512 | 512 | yes |
| register-blocked: q = 1, r = 4 | 4 | 1 | 16 | 256 | 256 | 512 | yes |
<!-- /worked -->

Two things are visible at once. The global loads, 256 in each of the last three rows, depend on $T$ alone, as the formula says, whichever way the workgroup splits it. The shared-memory reads fall with $r$: 1,024, 512, 256, which is $2n^3/r$ (the naive kernel
reads none, and pays for its loads in global memory). Nothing else changed: the 512 multiply-adds are the same, and so is the product.

What does the block buy? Two quantities fall. The global traffic falls with the tile side $T = 16r$ of a workgroup of 256 invocations, so that a block of $r = 8$ gives a tile of 128, eight times the 16 of the portable tiled kernel, and 16.8 million loads
instead of 134 million for $n = 1024$. And the shared-memory traffic per operation falls with $r$ itself, to $2/r$ numbers per multiply-add. That second saving matters because shared memory has a bandwidth too. In the model (a parameter of this chapter, as the 32 banks of 4 bytes
that real devices have) each compute unit's shared memory delivers 128 bytes per cycle, while its 128 lanes do 128 multiply-adds per cycle, 256 operations. A tile with $r = 1$ asks for 8 bytes of operands per multiply-add, 4 bytes per operation: $128/4 = 32$ operations per cycle can be fed, an eighth of the 256 the lanes can do. In general

$$
\begin{aligned}
\frac{\text{bytes of shared memory}}{\text{operation}} &= \frac{8r}{2r^2} = \frac{4}{r},\\[4pt]
\text{operations per cycle that shared memory can feed} &= 32\,r,
\end{aligned}
$$

and $r \ge 8$ is needed for the shared memory to keep up with the lanes (the argument of Volkov and Demmel for the register block of a GPU matrix product, restated in this chapter's model). This is a second roofline, with shared-memory bandwidth in place of device bandwidth and $4/r$ bytes per operation in place of $1/\mathcal{I}$.

The price is paid in registers. An invocation now holds $r^2$ accumulators and $2r$ operand registers, besides the registers for its indices and loop counters. In the model, $N_{\mathrm{reg}} = 32 + r^2 + 2r$, where the 32 is a parameter of this chapter. By Chapter 4 the registers
set the occupancy, and below about 500 resident invocations per compute unit the latency of the loads is no longer hidden.

<!-- worked: matmul_fast.register_blocking -->
C = A B, n = 1024, fp32, workgroup of 256 invocations (16 x 16), each computing an r x r block, so the workgroup tile is T = 16 r on a side; k-chunk 8, shared memory (T + T) x 8 x 4 bytes. PARAMETERS of this chapter: registers per invocation = 32 + r^2 + 2 r (32 for indices and loop counters, r^2 accumulators, 2 r operands); shared memory delivers 128 bytes per cycle per compute unit (20,480 (GB/s) over the 80 compute units at 2 GHz). Shared roof = shared bandwidth / (4 / r bytes per FLOP); global roof = intensity x 800; both in (GFLOP/s); the arithmetic peak is 40,000. Occupancy by the model of Chapter 4; 'enough' = at least 500 resident invocations per compute unit.

| r | tile T | registers per invocation | shared bytes | resident invocations | occupancy (percent) | enough | global loads | FLOP per byte | shared reads per FMA | global roof | shared roof | attainable (limit) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 35 | 1,024 | 1,792 | 87.5 | yes | 134,217,728 | 3.97 | 2.000 | 3,175 | 5,120 | 3,175 (global memory) |
| 2 | 32 | 40 | 2,048 | 1,536 | 75.0 | yes | 67,108,864 | 7.88 | 1.000 | 6,302 | 10,240 | 6,302 (global memory) |
| 4 | 64 | 56 | 4,096 | 1,024 | 50.0 | yes | 33,554,432 | 15.52 | 0.500 | 12,412 | 20,480 | 12,412 (global memory) |
| 8 | 128 | 112 | 8,192 | 512 | 25.0 | yes | 16,777,216 | 30.12 | 0.250 | 24,094 | 40,960 | 24,094 (global memory) |
| 16 | 256 | 320 | 16,384 | does not fit | - | - | - | - | - | - | - | - |
<!-- /worked -->

Read the rows downwards. The global loads fall with $T = 16r$ and the intensity climbs from 3.97 to 30.12. Shared-memory reads per multiply-add fall from 2 to $1/4$. The registers climb with $r^2$: 35, 40, 56, 112, and occupancy falls from 87.5 percent to 25 percent,
512 resident invocations, just above the 500 that Chapter 4 asks for. At $r = 16$ the invocation would need 320 registers, more than the 255 an invocation can name, and the kernel cannot run: the block is bounded above by registers. In the last column the global roof is the lower of the two in every row, so in this model the
shared-memory roof never binds, because the tile that makes it fit ($r \ge 8$) is also the tile that lifts the global roof. The choice $r = 8$ is what the model picks and it is what brain's register-tiled kernels use (the last section); both limits, 112 registers against the limit of 255 and shared-memory bandwidth against $r \ge 8$, bracket it.

The same trade exists on a CPU core. A core has registers too, and the CPU backend of brain applies the same idea there. A vector core with 16 vector registers, a fused multiply-add of latency 5 cycles started twice per cycle (Chapter 2) needs 10 independent accumulators to keep both ports busy, and
a block of $r_m \times r_n$ vector accumulators needs $r_m r_n$ accumulators, $r_m$ vectors of $A$ and one temporary for the vector of $B$ just loaded:

<!-- worked: matmul_fast.cpu_register_block -->
A core with 16 vector registers, a fused multiply-add of latency 5 cycles started 2 per cycle (Chapter 2), so 10 independent accumulators keep both ports busy. A block of r_m x r_n vector accumulators needs r_m r_n accumulators, r_m vectors of A and one temporary for the vector of B just loaded: r_m r_n + r_m + 1 registers; one k-step does r_m r_n fused multiply-adds and r_m + r_n vector loads. Blocks that fit in 16 registers (selected rows; the best fit by FMAs per load is 3 x 4):

| block r_m x r_n | registers | FMAs per k-step | loads per k-step | FMAs per load | at least 10 accumulators |
|---|---:|---:|---:|---:|---:|
| 3 x 4 | 16 | 12 | 7 | 1.714 | yes |
| 2 x 6 | 15 | 12 | 8 | 1.500 | yes |
| 4 x 2 | 13 | 8 | 6 | 1.333 | no |
| 2 x 2 | 7 | 4 | 4 | 1.000 | no |
| 1 x 4 | 6 | 4 | 5 | 0.800 | no |
| 1 x 1 | 3 | 1 | 2 | 0.500 | no |
<!-- /worked -->

The best block that fits is $3 \times 4$: 12 accumulators, 3 vectors of $A$ and 1 temporary, exactly 16 registers, with 12 multiply-adds per 7 loads. A $4 \times 2$ block fits easily and has too few chains to hide the latency of the multiply-add; a $2 \times 6$ block fits too but loads more per multiply-add.
The kernel `block3x4_abt` in `crates/backend-cpu/src/fast_ops.rs` has this block, and its comment states the same count.

## Shared memory has banks

Chapter 5 left a cost of shared memory open, the access pattern `tile[lx * T + ly]` of its transposing kernel. Reading a tile column by column is exactly what a matrix product does with $B$, so we count it here.

Shared memory is organized in **banks**: in the model, 32 banks of 4-byte words, with word $a$ in bank $a \bmod 32$. A bank serves one word per cycle. The 32 lanes of a subgroup issue one access together, and it takes as many cycles as the most distinct words that fall in a single bank; lanes
that read the same word are served together by a broadcast. That count is a **bank conflict** of that **degree** (degree 1 means no conflict). If lane $t$ reads word $s\,t$, a column of a row-major tile of row stride $s$, the banks it touches are $s t \bmod 32$,
and the degree is $\gcd(s, 32)$: a stride of 1 is conflict-free, a stride of 32 puts all 32 lanes on one bank. Padding an even stride by one word, to $s+1$, makes it odd and so conflict-free:

<!-- worked: matmul_fast.bank_conflicts -->
Shared memory of the model: 32 banks of 4-byte words; word a lives in bank a mod 32; the 32 lanes of a subgroup issue one access and it takes as many cycles as the most distinct words in one bank (lanes reading the same word are served together). 'Degree' is that count.

Lane t reads word `stride x t` (a column of a row-major tile):

| stride (words) | banks used by 32 lanes | degree |
|---|---:|---:|
| 1 | 32 | 1 |
| 2 | 16 | 2 |
| 4 | 8 | 4 |
| 8 | 4 | 8 |
| 16 | 2 | 16 |
| 32 | 1 | 32 |
| 33 | 32 | 1 |

The accesses of the tiles in this chapter (32 lanes unless the line says 16):

| access | degree |
|---|---:|
| tile[lx x 16 + ly], lanes l = 0 .. 31 | 8 |
| tile[lx x 17 + ly], lanes l = 0 .. 31 | 2 |
| B-tile read, thread tx owns 8 adjacent columns: Bs[kk x 128 + 8 tx + j], tx = 0 .. 15, j fixed | 4 |
| B-tile read, interleaved columns: Bs[kk x 129 + tx + 16 j], tx = 0 .. 15, j fixed | 1 |
| staging store As[kk x 128 + r], lanes idx = 0 .. 31 with r = idx div 8, kk = idx mod 8 (no padding) | 8 |
| staging store As[kk x 129 + r], lanes idx = 0 .. 31 with r = idx div 8, kk = idx mod 8 (padded by one word) | 4 |
| staging store As[kk x 132 + r], lanes idx = 0 .. 31 with r = idx div 8, kk = idx mod 8 (padded by one quad of four words) | 1 |
<!-- /worked -->

The first table is the rule. The second applies it to the accesses of this chapter. The transposing read of Chapter 5, 16 lanes across and two rows down, is an 8-way conflict at a row stride of 16 words and a 2-way conflict at 17 (not
1: lane 0 of row 0 and lane 15 of row 1 land on bank 0 again). In the register-tiled kernels, a thread that owns 8 adjacent columns of the $B$ tile reads words $8\,t_x + j$ for $t_x = 0 \ldots 15$, which fall on 4 banks, a 4-way conflict; giving thread $t_x$ the columns $t_x, t_x + 16, t_x + 32, \dots$ instead
makes the 16 reads consecutive words, one per bank. And the staging store, where consecutive invocations take consecutive $k$ of one row of the tile (so that their global loads coalesce) and write it $k$-major, is an 8-way conflict at a stride of 128 words, 4-way at 129 and conflict-free at 132. Padding and interleaving change nothing in the arithmetic, only the cycles of each shared access.

## How far each idea climbed

Each section above moved the product up the roofline of Chapter 3. To read how far each idea took it and what is left, we place every kernel on the model GPU with one computation. The kernels are the naive loop, the tiled kernels of the largest portable tile ($T = 16$) and of $T = 32$, and the register-blocked kernels of 16 by 16 invocations with $r = 4$ and $r = 8$, all on $n = 1024$ with fp32 operands. The table adds two rows for the matrix unit of the next section; read those once you have it.

<!-- worked: matmul_fast.roofline_climb -->
C = A B, n = 1024, on the model GPU: arithmetic peak 40,000 (GFLOP/s) (Chapter 3), bandwidth 800 (GB/s), ridge 50 FLOP per byte; the matrix unit's peak is 8 times the vector peak, 320,000 (GFLOP/s), a PARAMETER of this chapter, with ridge 400. Bytes: fp32 operands 4 (2 n^3 / T + n^2); matrix-unit rows read fp16 operands (2 bytes) and write the fp32 output. Attainable = min(peak, intensity x 800) with global traffic only, the model's roofline of Chapter 3; shared memory is not the limit in any row.

| kernel | FLOP per byte | attainable (GFLOP/s) | peak of its units (GFLOP/s) | fraction of that peak | side of its ridge |
|---|---:|---:|---:|---:|---|
| naive | 0.25 | 200 | 40,000 | 0.0050 | bandwidth |
| tiled, T = 16 | 3.97 | 3,175 | 40,000 | 0.0794 | bandwidth |
| tiled, T = 32 | 7.88 | 6,302 | 40,000 | 0.1575 | bandwidth |
| register-blocked, T = 64 | 15.52 | 12,412 | 40,000 | 0.3103 | bandwidth |
| register-blocked, T = 128 | 30.12 | 24,094 | 40,000 | 0.6024 | bandwidth |
| matrix unit, fp16, T = 128 | 56.89 | 45,511 | 320,000 | 0.1422 | bandwidth |
| matrix unit, fp16, T = 256 | 102.40 | 81,920 | 320,000 | 0.2560 | bandwidth |
<!-- /worked -->

![Log-log roofline of the model GPU: the rising line of slope one, meeting a flat vector roof at intensity 50 and a flat matrix-unit roof eight times higher at intensity 400. Green points for the naive kernel, T = 16, T = 32, T = 64 and T = 128 climb the sloped part from 0.25 to 30 operations per byte; two red points for fp16 operands lie beyond the vector ridge on the dashed matrix-unit roof.](img/matmul-roofline.svg)

The naive kernel sits at a fraction 0.005 of the peak, the tiled kernel at $T = 16$ at 0.079, the register-blocked kernels at 0.310 and 0.602. Each step moved the point to the right along the slanted part of the roof, 0.25, 3.97, 15.52, 30.12 operations per byte, and
the attainable rate rose in proportion. That is the success: 120 times the attainable rate of the naive loop, from reorganizing the same $2n^3$ operations. It also shows the model's limit. Even the best of the five kernels is left of the ridge point of 50, bandwidth-bound,
and by the formula the tile that reaches the ridge in fp32 would be $T = 4 \times 50 = 200$ on a side, beyond what 255 registers can hold.

Two things close that gap in practice and neither is in the roofline of the model. A real GPU has a cache between the compute units and device memory, so a tile loaded by one workgroup is often served to its neighbour, and the bytes that reach device memory are fewer than the global loads the model counts; the count here is of requests, as the closing paragraph of Chapter 4's coalescing section also noted. And the
roofline is a ceiling: how close a kernel gets depends on whether its instruction stream is mostly multiply-adds, on how well it hides the latency of its loads, and on the idle lanes of partial tiles. The section on brain gives an example of each, dated and with their setup.

## Tensor cores and cooperative matrices

The vector lanes of Chapter 4 do one multiply-add per lane per cycle, however cleverly they are fed. A GPU can have units that multiply a small matrix by a small matrix in one operation, and the matrix product is the computation they exist for. We count what one such operation does, and what it changes in the roofline and in the program.

A **tensor core** (the vendors' names for the idea differ, and the book uses this one) is a hardware unit that multiplies two small fixed-size matrices and adds a third in one operation: $D = AB + C$ with $A$ of $m_f \times k_f$, $B$ of $k_f \times n_f$ and $C$, $D$ of $m_f \times n_f$. A shape of
$16 \times 16 \times 16$ is common. The inputs are usually in a narrow format (fp16 here) and the sum is accumulated in fp32, the arrangement that Chapter 1 gave a reason for: a product of two fp16 numbers has a significand of at most 22 bits, which fp32's 24 hold exactly, and an fp32 accumulator does not stall as an fp16 one does.
One operation does $2\,m_f n_f k_f = 8{,}192$ operations. A subgroup of 32 lanes issuing an ordinary multiply-add does 64. The next block runs the product of two $32 \times 32$ matrices in $16 \times 16 \times 16$ fragments and counts both ways:

<!-- worked: matmul_fast.fragment_product -->
A and B are [32, 32] with entries `2 uniform() - 1` rounded to float32 and then to fp16 (round to nearest, ties to even) for the fragment path. Generator: the book's SplitMix64, seed 0x222e1faaed445854 (the seed of the name `matmul_fast.fragment_product`). Draw order: the 1024 entries of A row by row, then the 1024 of B row by row. A fragment multiply-add forms the products exactly and rounds the sum with the accumulator once to float32 (the book's model of a fragment unit). The product is cut into 16 x 16 output fragments and 2 steps along k.

| quantity | value |
|---|---:|
| fragment shape m_f x n_f x k_f | 16 x 16 x 16 |
| FLOPs of one fragment multiply-add 2 m_f n_f k_f | 8,192 |
| fragment multiply-adds for [32, 32] by [32, 32] | 8 |
| FLOPs of the product 2 n^3 = fragment multiply-adds x FLOPs each | 65,536 |
| subgroup instructions of vector fused multiply-adds (32 lanes) for the same product | 1,024 |
| accumulator registers per lane, m_f n_f / 32 | 8 |
| numbers loaded from A and B = fragment multiply-adds x (m_f k_f + k_f n_f) = 8 x (256 + 256) | 4,096 |
| largest |D - exact product of the fp16 inputs| (accumulation only) | 2.701e-07 |
| largest |D - exact product of the original fp32 inputs| (inputs rounded to fp16) | 1.726e-03 |
| largest |sequential fp32 loop - exact product of the fp32 inputs| | 2.190e-06 |
<!-- /worked -->

The product needs eight fragment operations (2 by 2 output fragments, each accumulating 2 steps along $k$) where the lanes would need 1,024 subgroup instructions, 128 times as many; the ratio is $8192/64$. The result is not the same number as the fp32
product, and the table says where the difference is. Against the exact product of the fp16-rounded inputs, the fragment result differs by $2.7 \times 10^{-7}$, the rounding of the accumulation, an fp32 effect. Against the exact product of the original fp32 inputs it differs by up to $1.7 \times 10^{-3}$,
the rounding of the inputs to fp16, whose relative error $2^{-11}$ is $2^{13} = 8192$ times the unit roundoff of fp32, and a sum of 32 such terms keeps a good part of it. A sequential fp32 loop is within $2.2 \times 10^{-6}$ of the exact product.
This is the first-order rule of Volume I: each rounding contributes a relative error of at most the unit roundoff of the format it happens in, so rounding the *inputs* to fp16 contributes up to about $2 \times 2^{-11}$ of $S = \sum_k |a_{ik} b_{kj}|$ (the sum of the magnitudes of the terms) whatever the accumulator does, while the accumulator's own roundings are of order $2^{-24}$ per term. The precision of real tensor cores is the subject of Markidis and colleagues' study; here the speed is bought with the width of the inputs, and the accumulator is where the book spends bits to keep that error from growing.

The speed of the unit is the other half of the story, and it is a parameter of this chapter, in the way Chapter 3's machines are: a device whose matrix units complete eight times the operations per cycle of its vector lanes, $F_{\mathrm{tc}} = 8 \times 40000 = 320000$ (GFLOP/s) (real ratios on real devices differ and are a datasheet property
of a particular part; this one is a model). Its ridge point against the same 800 (GB/s) is $320000/800 = 400$ operations per byte, eight times higher than the vector ridge of 50, so a kernel that was compute-bound on the vector lanes is bandwidth-bound on the matrix unit unless it moves less. The operands help: fp16 halves their bytes, so
the intensity of a tile of side $T$ is about $T/2$ instead of $T/4$.

<!-- worked: matmul_fast.tile_for_ridge -->
For large n the global traffic is 2 n^3 / T operand reads, so intensity -> T / (operand bytes) FLOP per byte (the writes of C vanish). To reach a ridge the workgroup tile must have side T = ridge x operand bytes.

| units | ridge (FLOP per byte) | operand bytes | tile side T to reach it | next power of two |
|---|---:|---:|---:|---:|
| vector units, fp32 operands | 50 | 4 | 200 | 256 |
| matrix unit, fp16 operands | 400 | 2 | 800 | 1024 |
<!-- /worked -->

The rows of the previous roofline table for the matrix unit agree: a tile of 128 with fp16 operands has an intensity of 56.89, 0.142 of the matrix unit's peak, and a tile of 256 reaches 102.40, 0.256, both bandwidth-bound by a wide margin and both above the whole vector peak of 40,000 (GFLOP/s) (the unit helps even where it is starved). To reach the matrix unit's ridge in the model a workgroup tile would need a side of 800, which no register file or
shared memory holds. Real kernels bridge it by the means of the last section's cache between workgroups, by larger tiles with narrower operands (fp8 and int8 halve the bytes again), and by a hierarchy of tiles, a workgroup tile made of warp tiles made of fragments. The qualitative reading is the one to keep: the faster the units, the more reuse the program must arrange, and matrix units
raise the bar for reuse eightfold at the same bandwidth.

A fragment is a matrix, but not one that any single invocation holds. The 256 elements of a $16 \times 16$ accumulator are spread over the 32 lanes of a subgroup, 8 registers per lane, and the program may not say which lane has which element. A **cooperative matrix** is that: a small matrix whose elements are held jointly by the invocations of one subgroup,
operated on by subgroup-wide operations (load a fragment from memory, multiply-add fragments, store a fragment), each executed by all the lanes together. The pseudocode of the book has four such operations (defined in `docs/book/pseudocode.md`). The kernel with one subgroup per output fragment:

```text
kernel matmul_coop(A: array<f16>[M, K], B: array<f16>[K, N], C: inout array<f32>[M, N], M: int, N: int, K: int) workgroup n_sg:
    require M mod 16 == 0 and N mod 16 == 0 and K mod 16 == 0
    acc = fragment_zero(f32, 16, 16)
    for t in 0 .. K // 16:
        fa = fragment_load(A, w_y * 16, t * 16, 16, 16)
        fb = fragment_load(B, t * 16, w_x * 16, 16, 16)
        acc = fragment_mad(fa, fb, acc)
    fragment_store(acc, C, w_y * 16, w_x * 16)

launch matmul_coop(A, B, C, M, N, K) over N // 16 by M // 16 workgroups
```

It is shorter than every kernel before it, because the arithmetic, the register block and the hand-off between lanes are all inside `fragment_mad`. What it does not contain is also a statement: there are no per-element guards (a fragment load reads a whole block, so a shape that is not a multiple of the fragment must be padded, and
the padding is a cost, like the idle rows of a partial tile in the next table), no explicit shared memory (each fragment is loaded straight from memory here, a deliberate simplification; a fast kernel stages tiles in shared memory exactly as in the previous sections and loads fragments from there), and no control of the order of the additions inside a fragment.

<!-- worked: matmul_fast.tile_fill -->
Output [m, 2560] from a 128 x 128 workgroup tile (n = 2560 is exactly 20 tiles wide) and from 16 x 16 fragments (rows padded to a multiple of 16; 2560 needs no padding). Share of the computed outputs that are real.

| m | 128 x 128 workgroups | real share of tile outputs (%) | m padded to 16 | real share of fragment rows (%) |
|---|---:|---:|---:|---:|
| 1 | 20 | 0.8 | 16 | 6.2 |
| 8 | 20 | 6.2 | 16 | 50.0 |
| 32 | 20 | 25.0 | 32 | 100.0 |
| 77 | 20 | 60.2 | 80 | 96.2 |
| 128 | 20 | 100.0 | 128 | 100.0 |
| 129 | 40 | 50.4 | 144 | 89.6 |
| 512 | 80 | 100.0 | 512 | 100.0 |
<!-- /worked -->

Why must this path be gated on a capability? Chapter 4 explained a subgroup, and brain's kernels, written in the portable subset, do not use subgroup operations at all: they are core compute that runs on any device that runs the language. A cooperative matrix cannot follow that rule, because it is by definition an operation among the lanes of a
subgroup, with a shape and an element type that the hardware fixes. Devices differ in which shapes and which type triples (fp16 inputs with fp32 accumulation, int8 inputs with int32 accumulation, and so on) they support, and many support none. So the path is an optional one: the program has to ask the device which fragment shapes and types it has,
and run the fragment kernel only if the answer matches what the kernel was written for; otherwise it falls back to the portable kernels, whose results the fast path is checked against. The question "does the device have it" is answered by a query, not by whether compiling the kernel happens to succeed, and the last section shows a case where the difference mattered.

## Which rearrangements change a sum

Every rearrangement above reorders arithmetic. In exact arithmetic that is invisible; in floating point it is not, so we find which reorderings change a result, and by how much, in order that a test compares what it should compare.

Volume I, Chapter 15 showed that floating-point addition is not associative: $(1 + 2^{-24}) + 2^{-24} = 1$ in fp32 and $1 + (2^{-24} + 2^{-24}) = 1 + 2^{-23}$. A product $C_{ij}$ is a sum of $K$ terms, so a kernel that adds them in another order may return another number in the last bits. Which rearrangements do?

A **tiled** or **register-blocked** kernel, as written above, does not. Each output has one accumulator, which a workgroup carries from step to step, and the steps and the inner loop both run in ascending $k$: the additions of $C_{ij}$ happen in the order $0, 1, \ldots, K-1$, as in the naive loop. Tiling changes where the numbers come from and which invocation
holds them, never the order in which one output sums its terms. What does change the order is a design that splits the sum: a partial sum per tile added to the total, the **split-K** schedule that cuts $K$ into slices for several workgroups and adds the partial results afterwards, the several accumulators of a vector core, and a fused multiply-add (one rounding per term instead of two). A fragment multiply-add sums $k_f$ products at a
time, in an order the hardware chooses, with inputs rounded to a narrower format. The next table adds the same 1024 products of 16 seeded outputs in seven ways:

<!-- worked: matmul_fast.summation_orders -->
C = A B with A [4, 1024] and B [1024, 4], entries `2 uniform() - 1` rounded to float32. Generator: the book's SplitMix64, seed 0xf5470b48859cbe35 (the seed of the name `matmul_fast.summation_orders`). Draw order: the 4096 entries of A row by row, then the 4096 entries of B row by row. Every product and every sum is a correctly rounded float32 operation (no fused multiply-add) unless the row says otherwise. With p = 0 .. 1023 the index of the sum and fl(.) a rounding to float32, the seven orders are: sequential, acc = fl(acc + fl(a_p b_p)) for p ascending from acc = 0; tiled with the accumulator carried, the same additions in the same order written as 64 tiles of 16 consecutive p (p ascending inside a tile, one accumulator for the whole sum); fused multiply-add, acc = fl(acc + a_p b_p) with the product exact (one rounding per term); split-k with 4 slices, the four slices of 256 consecutive p are each summed sequentially from 0 (rounded products) and the four slice sums are then added sequentially, slice 0 first, from acc = 0; tile sums added to the total, each of the 64 tiles of 16 consecutive p is summed sequentially from 0 and the tile sums are then added sequentially to acc, tile 0 first; 8 vector lanes, lane j = fl(lane j + fl(a_p b_p)) for the p with p mod 8 = j in ascending order, then s_j = fl(lane j + lane j + 4) for j = 0 .. 3 and the result fl(fl(s_0 + s_1) + fl(s_2 + s_3)); fragments with fp16 inputs, a and b first rounded to fp16, then for each tile of 16 consecutive p the 16 products are formed exactly, added exactly to acc, and the sum is rounded once to float32. 'Differ' counts the outputs whose float32 value differs from the sequential order. Error is |result - exact sum of the float32 inputs|, with the exact sum computed in rational arithmetic, divided by u S, where u = 2^-24 and S = the sum of |A[i, p] B[p, j]| (Volume I's bound for a sum of K terms is (K - 1) u S = 1,023 u S).

| order of the sum | differ from sequential | largest error / (u S) | mean error / (u S) |
|---|---:|---:|---:|
| sequential | 0 of 16 | 0.846 | 0.257 |
| tiled, accumulator carried | 0 of 16 | 0.846 | 0.257 |
| fused multiply-add | 15 of 16 | 0.972 | 0.299 |
| split-k, 4 slices | 15 of 16 | 0.291 | 0.188 |
| tile sums added to the total | 13 of 16 | 0.192 | 0.074 |
| 8 vector lanes | 16 of 16 | 0.247 | 0.094 |
| fragments, fp16 inputs | 16 of 16 | 374.340 | 139.064 |

The output C[0, 0] under each order: its value to 9 significant digits, its float32 bit pattern in hexadecimal, and its distance from the sequential result in units of the float32 spacing there (ulp).

| order of the sum | C[0, 0] | bits | ulp from sequential |
|---|---:|---:|---:|
| sequential | -8.10983467 | c101c1e2 | +0 |
| tiled, accumulator carried | -8.10983467 | c101c1e2 | +0 |
| fused multiply-add | -8.10983658 | c101c1e4 | -2 |
| split-k, 4 slices | -8.10983181 | c101c1df | +3 |
| tile sums added to the total | -8.10983276 | c101c1e0 | +2 |
| 8 vector lanes | -8.10983181 | c101c1df | +3 |
| fragments, fp16 inputs | -8.10430717 | c101ab3e | +5796 |
<!-- /worked -->

The "tiled" row, written as a different loop (tile loops outermost, ascending $k$ inside) with its accumulator carried, equals the sequential row in all 16 outputs, bit for bit. Every other fp32 order differs from it in 13 to 16 outputs of 16, in the last bits: for $C_{00}$ the fused multiply-add, the
split-K schedule and the vector lanes land 2, 3 and 3 units in the last place from the sequential value, with the bit patterns shown. Their errors against the exact sum are all of the same order, a fraction of $uS$, where $S = \sum_k |a_{ik} b_{kj}|$ is the sum of the magnitudes of the terms and $u = 2^{-24}$: the worst error of the sequential order is $0.846\,uS$, of the vector lanes $0.247\,uS$, and of
tile sums added to the total $0.192\,uS$. Volume I's bound for a sum of $K$ terms is $(K-1)\,uS = 1023\,uS$; every error in the table is below $1\,uS$, a thousand times under the bound, since the individual roundings are of both signs. The last row is of another size: with fp16 inputs the largest error is $374\,uS$, the error of the inputs, not of the sum.

Three consequences follow. A test of a tuned kernel against a reference needs a tolerance sized by $uS$ (or by the unit roundoff of the *input* format when the inputs are narrowed), unless the kernel is meant to keep the order, in which case an exact comparison is a stronger test and brain makes it (the last section). A comparison with a tolerance
should not be tightened to the zero that happened to come out: that zero is a property of today's order of additions, and the next legitimate reordering changes the last bits, so a test demanding zero would then report a regression where there is none (knowledge #119 records the same lesson). And the order of additions is itself a property worth recording: a kernel that is reproducible from run to run needs a fixed order, which split-K and a tree
reduction can have, as long as the slices are folded in a fixed sequence, and which an unordered accumulate (an atomic add) cannot give.

## The matmul ladder in brain's kernels

The matmul kernels of brain live in `crates/kernels/wgsl/`, one file per variant, and between them they are the chapter's ladder.

**The ladder, as files.** `matmul.wgsl` is the naive kernel: a header that says "one thread per output element, serial inner reduction", `@workgroup_size(64)`, a plain fp32 loop over $K$, and the output convention $out = x W^{\top}$ of a linear layer. `matmul_tiled.wgsl` is the first tiling:
a 32 by 32 output tile, 64 invocations in an 8 by 8 grid, each owning a $4 \times 4$ block of outputs and a step depth of 8, with two barriers per step; its header counts the intensity at 8 operations per byte. `matmul_reg.wgsl` is the register-blocked kernel: a $128 \times 128$ tile, 256 invocations in a $16 \times 16$ grid, each holding an $8 \times 8$ block, 8 KiB of workgroup memory for the two tile slabs,
and an intensity of 32 per its header (the model above gives 30.12 once the stores of $C$ are counted). Its accumulators are 64 separate scalar variables, not an array, and the header says why: an array indexed by the loop is placed in local memory, backed by device memory, and a product whose accumulators live there runs at the speed of that memory whatever
its tiling. `matmul_tile16.wgsl` is the exact small-tile kernel of $16 \times 16$ outputs with one output per invocation and step depth 64, its tile padded by one word per row to avoid the bank conflicts of the previous section. Which kernel a model runs is chosen by shape (the next paragraph). Their tile sizes are not free constants: `crates/backend-api/src/select.rs`
describes a `Schedule` of a split-K factor and a `TileShape` (`Wide128` or `Narrow64`), and `matmul_reg3_64.wgsl` is the same kernel retiled to $64 \times 64$ with $4 \times 4$ blocks, which exists, in its header's words, to waste fewer lanes on a tile that a short matrix leaves half empty.

**Choosing the variant.** `candidates` in `crates/backend-api/src/select.rs` returns, for a matrix product, an ordered list of `KernelVariant`s from the shape and the device: the one-workgroup-per-output decode kernel for a few rows, `RegisterTiled` once a tile is fillable ($m \ge 8$ rows and $n \ge 128$ columns, `GEMM_TILE_MIN_ROWS` and `GEMM_TILE_MIN_COLS`),
and the naive `Reference` always last, since it is the kernel everything else is checked against; `RegisterTiled` carries a `Requirement` for the device's workgroup-barrier capability, because the kernel stages its tile behind barriers. `pick_gemm` in `crates/model/src/block.rs` turns the choice into a dispatch whose unit is a workgroup per output tile, $\lceil m/128 \rceil \lceil n/128 \rceil$ of them. The threshold 8 is the measured crossover of knowledge #15,
the story chapter 4 told: a selection rule that sent a short matrix to the naive kernel cost more than the idle rows of a partial tile (the table of idle rows in the section on cooperative matrices gives the real share of a $128 \times 128$ tile: 60.2 percent at $m = 77$). The kernel `matmul_reg3_splitk.wgsl` is split-K, with the partial results folded by `dw_splitk_reduce.wgsl`, which is
also the kernel of chapter 5's grid-folding incident. The test `pick_gemm_routes_by_the_measured_crossover` in `crates/model/src/block.rs` pins the thresholds, and `crates/glmdsa/tests/reg3_parity.rs` checks that the register-tiled kernel reproduces the naive kernel bit for bit on a real model forward, the order claim of the previous section made executable.

**Bank conflicts, and a claim that was half right: commits 455f8054a, 9e03b6017 and c920a7d51.** `matmul_reg2.wgsl` stages tiles in shared memory with the layout of `matmul_reg.wgsl`. Commit 455f8054a (July 2026) added `matmul_reg3.wgsl` with two layout changes and nothing else: interleaved ownership of rows and columns
(thread $t$ owns $t, t+16, t+32, \ldots$) and a tile stride of 129 instead of 128. The commit records that the order of accumulation was unchanged and that the output was bit-identical to reg2's (a largest absolute difference of exactly zero over twelve shapes). Commit 9e03b6017 (August 2026) swept twelve shapes again and found reg3 the faster at every one, with the same zero difference, and found the consequence of
having kept both kernels: the UNet's linears were dispatched to the slower one, though the faster was registered in the same process. Commit 9b6415a6c, the same day, switched the crates that still selected reg2 (its title counts eleven). Knowledge #17 states the rule: when two kernels differ only by an optimization, the slow one should not be selectable. The bank table of this chapter's `bank_conflicts` block is the check of a claim in that story. The header of `matmul_reg3.wgsl`
says the padded stride removes the conflict of the staging store. The table computes degree 4 for stride 129, down from 8, not 1, and commit c920a7d51 (September 2026) says the same from the other direction: reg3's stride leaves a 4-way conflict, and its new kernel `matmul_reg4.wgsl` pads the tile to a stride of 132 words, the conflict-free row of the table. That commit also
corrects its own first reasoning: an argument that vectorized shared-memory reads would multiply shared bandwidth fourfold was wrong, because the shared-memory units are balanced against the banks, and what the wider reads save is instruction issue slots (the accounting of Lai and Seznec for matrix products on GPUs). The kernel's header carries the corrected reasoning and the register-tile bound of this chapter: the shared-memory bytes per operation, $4/r$, against 0.5 available, force $r \ge 8$. Its numbers are of one setup:

<!-- perf-number: historical measurement, commit c920a7d51 -->
In September 2026, on a Tesla P40 of datasheet peak 11,760 (GFLOP/s), through the native Vulkan backend, with the arms interleaved and the minimum of five runs, the commit reports that `matmul_reg3` reached about 41 percent of that peak on a square product of side 2048, `matmul_reg4` 49.8 percent, and 58.5 percent at the widest prefill shape it measured.

The ledger entry of commit 67e70ff72 records that a literature-grounded ceiling for a hand-written WGSL kernel on that part is 40 to 65 percent of the fp32 peak, and attributes the remaining headroom to what the language cannot control, not to the algorithm. Our roofline's 0.602 for the $128$ tile is a ceiling of a different kind (bytes only); the measured fractions are lower because of everything the roofline omits.

**The CPU backend: commit 483979045.** The same reuse problem on the CPU backend of brain. `matmul_abt` in `crates/backend-cpu/src/fast_ops.rs` walked the whole of $B$ once per row of $A$, so a thread with $R$ rows moved $R n k \cdot 4$ bytes, the ikj-style stream of this chapter's loop table; the commit's own account is that it was bound by memory bandwidth and not by arithmetic. It inverts the nest to hold a tile of four columns of $B$ resident and sweeps the
thread's rows through it, which drops the traffic to $n k \cdot 4$ per thread, and computes each tile with the $3 \times 4$ register block counted above. The choice of nest is by shape: it is not applied when $B$ is the small operand, and the commit records both directions measured.

<!-- perf-number: historical measurement, commit 483979045 -->
In September 2026, on a host of two 12-core Xeon processors with AVX2 and fused multiply-add but no AVX-512, the commit measured the feed-forward product of its model at 132.5 then 304.5 (GFLOP/s) before and after the change.

**Cooperative matrices: the gate, commits f1228f6c6 and 4847453f8.** The one hand-written non-WGSL matmul is `crates/vulkan/src/shaders_vk/matmul_coopmat.comp`, a GLSL kernel for `VK_KHR_cooperative_matrix` with $16 \times 16 \times 16$ fp16 fragments, one subgroup per output fragment and an fp32 accumulator, the pseudocode of this chapter nearly line for line;
WGSL as the book and brain use it has no cooperative-matrix construct, so this path cannot be a WGSL kernel. The capability query is in `crates/vulkan/src/context.rs` (`CoopMatCaps::supports_f16_tensorcore`: the feature is on and at least one enumerated shape matches fp16, fp16, fp32 at subgroup scope), and `crates/backend-vulkan/src/lib.rs` copies the enumerated shapes into the device description of `crates/backend-api/src/arch.rs` (`ArchDesc::matrix`).
The gate is the `Requirement.matrix` of `crates/backend-api/src/select.rs`, a triple of types that `satisfied_by` matches against the shapes the device enumerated; `CoopMatProvider` in `crates/gpu-core/src/provider/coopmat.rs` reports that requirement, and the registry checks it before the provider is ever asked to accept a request. The provider accepts only a forward product, with both operands already fp16 and all
three extents multiples of 16, and refuses a backward pass unconditionally: its documentation says fp16 products are not faithful enough for a gradient check (consistent with the rounding table above). Commit f1228f6c6 (September 2026), which added the provider, says that a dispatch of the kernel could not be checked on the machine it was written on, which has no cooperative-matrix shapes, and that the provider is therefore shown unreachable there by a test (`crates/gpu-core/tests/it/coopmat_provider_declines_on_this_box.rs`), not assumed to be.

That test found something recorded in its header. On the development machine's driver, creating a compute pipeline from the cooperative-matrix SPIR-V succeeded, although the device enumerates no cooperative-matrix shapes. A driver can accept a pipeline it would not execute correctly, so a successful registration is not evidence of capability, and the gate is the queried requirement and nothing else. Commit 4847453f8 says in the same words that its kernel changes were
unvalidated against a real dispatch for want of hardware. It (September 2026) moved the pipeline onto the main Vulkan backend's own device, and fixed a numeric bug in the packing of fp16 inputs: the earlier hand-written conversion flushed every fp32 value below the smallest normal fp16 number, $2^{-14}$, to zero, discarding the fp16 subnormals down to $2^{-24}$; it now uses a correctly rounding conversion, and two tests (`pack_f16_preserves_an_f16_subnormal_the_old_conversion_flushed_to_zero` and
`pack_f16_preserves_the_smallest_f16_subnormal` in `crates/backend-vulkan/src/coopmat.rs`) check that $2^{-15}$ and $2^{-24}$ survive. It is the narrowing of Chapter 1 doing its damage quietly: the small weights of a layer are exactly the ones that a flush to zero removes.

## Check your understanding

1. The loop-order table has the same 56,448 lines for ijk and ikj at $n = 96$ and very different counts at $n = 128$ with an 8-way cache. Explain both observations, and say what a third nest must do to beat them.
2. A tiled kernel is written with a barrier after the loads and none after the multiply-adds. It passes a test with $K = 16$, $T = 16$ and fails with $K = 64$. Explain.
3. A workgroup has $q = 8$ and $r = 2$, so 64 invocations, and a step depth of 8. Find its tile side, its registers per invocation in the model, its shared-memory bytes, its global loads for $n = 1024$ and its shared-memory reads per multiply-add, and compare each with the tiled kernel with $T = 16$.
4. A colleague proposes $r = 16$ "to halve the global loads once more". Use the model to say whether it can run, and what would have to change.
5. Row stride 130 words, rather than 128 or 129, is proposed for the tile that the staging store writes. Using the table's rule, say whether it removes the conflict. Why is 132 better?
6. A device has a matrix unit with eight times the vector peak. A kernel with a tile of 128 and fp16 operands has the intensity 56.89 of the table. Will it reach the peak of the matrix unit, and why does a faster unit make reuse more important?
7. Find the error: "tiling reassociates the sum, so the tiled kernel's output differs from the naive kernel's in the last bits, and a bitwise comparison of the two is a bug in the test."
8. A cooperative-matrix pipeline compiles and runs without an error on a device whose capability query lists no fragment shapes. A developer concludes that the capability check is unnecessary. What is wrong?

## Summary

The three-line loop for $C = AB$ performs $2n^3$ operations and loads $2n^3$ numbers, so its intensity is $1/(4 + 2/n)$, about a quarter of an operation per byte at every size, a fraction 0.005 of the peak of the model GPU, while the product itself can reach $n/6$. The gap is reuse: each of the $2n^2$ inputs is needed $n$ times
and the loop loads it each time. On a CPU, a blocked loop nest raises the reuse at the cache and cuts the lines fetched from 56,448 to 4,608 at $n = 96$. On a GPU, a workgroup computes a tile of $C$, stages tiles of $A$ and $B$ in shared memory and uses each loaded number $T$ times: the global loads are $k(m\lceil n/T\rceil + n\lceil m/T \rceil)$, $2n^3/T$ for a square product, and the intensity is about $T/4$, independent of the step depth. One output per invocation caps $T$ at 16 on a portable
workgroup of 256 invocations, so each invocation computes an $r \times r$ block in registers, lifting the tile to $16r$, cutting the shared-memory reads to $2/r$ per multiply-add (which makes $r \ge 8$ necessary to keep the lanes fed) and costing $r^2$ registers, which bound $r$ by the register file and set the occupancy. Shared-memory banks add a cost of their own, the degree of a bank conflict, removed by padding and interleaving without any change of the arithmetic. These steps took the model kernel
from a fraction 0.005 of the peak to 0.602.

A tensor core multiplies fragments in one operation, 8,192 operations for a $16 \times 16 \times 16$ fragment, with narrow inputs and an fp32 accumulator; in the model it lifts the peak eightfold and the ridge point to 400, so it demands still more reuse, and its programming model, a cooperative matrix held jointly by a subgroup, depends on a subgroup capability and a device-specific shape and type, so a program must query for it and keep the portable kernels as the fallback. The order of the additions is preserved by tiling and register
blocking with a carried accumulator, bit for bit, and changed by split-K, fused multiply-add, several accumulators and fragments, all within a small multiple of $uS$ except the narrowing of the inputs to fp16, which costs about 374 $uS$ in the seeded example. In brain the ladder is `matmul.wgsl`, `matmul_tiled.wgsl`, `matmul_reg*.wgsl`, selected by shape in `crates/backend-api/src/select.rs`, with a capability-gated cooperative-matrix path beside them.

What the chapter leaves is everything around the product. A network is not one matrix product: it is products, additions, reductions and normalizations, each a kernel with its own traffic. The next chapter takes up the reductions, which a product also needs at its edges and which the parallel machine handles differently.

## References

- Kazushige Goto and Robert A. van de Geijn, "Anatomy of high-performance matrix multiplication", 2008.
- Vasily Volkov and James W. Demmel, "Benchmarking GPUs to tune dense linear algebra", 2008.
- Junjie Lai and Andre Seznec, "Performance upper bound analysis and optimization of SGEMM on Fermi and Kepler GPUs", 2013.
- Stefano Markidis, Steven Wei Der Chien, Erwin Laure, Ivy Bo Peng and Jeffrey S. Vetter, "NVIDIA Tensor Core programmability, performance and precision", 2018.
- Nicholas J. Higham, "Accuracy and Stability of Numerical Algorithms", second edition, 2002.

## Exercises

1. **[Calculate]** For $C = AB$ with $A$ of $[2048, 1024]$ and $B$ of $[1024, 512]$, find the global loads of the naive kernel and of the tiled kernel with $T = 16$ from the exact formula, and their ratio.
2. **[Calculate]** For $q = 8$, $r = 4$ (64 invocations per workgroup), a step depth of 8 and $n = 1024$, find the tile side, the registers and shared-memory bytes per the model, the resident workgroups and occupancy by the model of Chapter 4, the global loads, and the shared-memory reads per multiply-add.
3. **[Calculate]** Find the bank-conflict degree of a column read with row strides 12, 24, 40 and 48 words, and the smallest padding that makes each conflict-free.
4. **[Calculate]** A core with 32 vector registers: using the register count of the CPU section, find the block $r_m \times r_n$ that maximizes multiply-adds per load, with its registers and ratio.
5. **[Calculate]** For fragments of $16 \times 16 \times 16$, find the fragment operations and the FLOPs of $[64, 80]$ by $[80, 48]$, then of $[64, 80]$ by $[80, 50]$ after padding to multiples of 16, and the share of the computed outputs that are real.
6. **[Calculate]** On the model GPU, find the intensity of the register-blocked kernel with $T = 128$ for $n = 4096$, the attainable rate and its fraction of the peak, and compare with $n = 1024$ (30.12). Explain the direction of the change from the formula.
7. **[Implement]** Reproduce `matmul_fast.loop_order_misses` with your own implementation of the cache of Chapter 2 and of the three nests, and add $n = 112$ (a row of 448 bytes, 7 lines). Report the three counts for both associativities and say which of the effects of the table (compulsory only, stream of $B$, set conflicts) each shows.
8. **[Implement]** Reproduce `matmul_fast.tiled_trace` and `matmul_fast.blocked_trace` with a simulator of your own for the workgroup kernel (phases separated by barriers, a count of each global load and shared read). Add the product $[10, 9]$ by $[9, 7]$ with $q = 2$, $r = 2$ and a step depth of 3, and compare the counted loads with the exact formula.
9. **[Implement]** Reproduce `matmul_fast.summation_orders` with the generator and draw order printed with it, and add a pairwise order that adds the products in a balanced binary tree. Report its differences and errors in the same units.
10. **[Implement]** Reproduce `matmul_fast.fragment_product`: round the inputs to fp16 with a correctly rounding conversion, implement the fragment multiply-add with one rounding of the exact sum, and report the three errors. Then change the model so that the fragment sum is rounded to fp32 after each of the 16 products and report how the first error changes.
11. **[Investigate]** Query your device (or your CPU) for its matrix-unit capability: the supported fragment or matrix shapes and element types, or their absence. State the interface and driver, whether the capability is optional, and what a program should do when it is missing.
12. **[Investigate]** Write the tiled kernel for a CPU or a GPU available to you and measure its bytes with your platform's counters, or count its cache-line fetches as the chapter's simulator does. Report the fraction of your machine's measured roofline for $n = 512$ and $n = 1024$ with the naive loop and the tiled one, stating the machine, compiler, repetitions and minimum, and explain any fraction above 1.

## Key terms

- **naive matmul**: the matrix product computed as its definition reads, one output at a time as a dot product of a row and a column, loading both from memory for every output and keeping nothing between outputs; its intensity is about a quarter of an operation per byte at any size. <!-- concept: naive-matmul -->
- **tiled matmul**: the matrix product computed one tile of the output at a time, with the matching tiles of the two inputs staged in a fast memory so that each number loaded is used once per row or column of the tile, cutting the loads from $2n^3$ to $2n^3/T$ for a tile of side $T$. <!-- concept: tiled-matmul -->
- **register blocking**: giving each invocation of a kernel a small square block of outputs held in its registers, so that each operand it reads is used once per row or column of the block, which enlarges the output tile beyond the number of invocations and cuts the reads of shared memory per multiply-add. <!-- concept: register-blocking -->
- **tensor core**: a hardware unit that multiplies two small fixed-size matrices and adds a third in one operation, usually with narrow inputs and a wider accumulator, raising the peak arithmetic rate of a device for matrix products. <!-- concept: tensor-core -->
- **cooperative matrix**: a small matrix whose elements are held jointly by the invocations of one subgroup and operated on by subgroup-wide load, multiply-add and store operations; the programming interface to tensor cores, available only where the device reports the shapes and types it supports. <!-- concept: cooperative-matrix -->

## Answers

**Recall questions.**

1. $2n^3 = 2 \times 512^3 = 268{,}435{,}456$ operations; at least $12 n^2 = 12 \times 262{,}144 = 3{,}145{,}728$ bytes; intensity $268{,}435{,}456 / 3{,}145{,}728 = n/6 = 85.33$. (Volume IV, Chapter 3.)
2. 256 lines, one per row (a row of 256 numbers is 1,024 bytes, 16 lines, and the column takes one number from one line of each row); one number of 16 per line. (Volume IV, Chapter 2.)
3. The invocations that read the tile use numbers written by other invocations of the workgroup. Without a barrier nothing orders a write before a read, so a read can return a stale value; the barrier makes every write finish before any read begins. (Volume IV, Chapter 4.)
4. $\lfloor 65536 / (40 \times 256) \rfloor = 6$ from registers, $\lfloor 65536 / 2048 \rfloor = 32$ from shared memory, $2048/256 = 8$ from slots, cap 16: 6 workgroups, 1,536 invocations, 75 percent. (Volume IV, Chapter 4.)
5. No. $1 + 2^{-24}$ is a tie between 1 and $1 + 2^{-23}$ and rounds to even, 1, so the first sum is $1 + 2^{-24} \to 1$, then $1 + 2^{-24} \to 1$: the result is 1. The second adds the two small numbers first, $2^{-23}$, and $1 + 2^{-23}$ is representable. A program that adds the same numbers in another order can therefore return another result, in the last bits. (Volume I, Chapter 15.)

**Check your understanding.**

1. At $n = 96$ the matrix $B$ (36 KiB) no longer fits the 32 KiB cache and both orders stream all of $B$ through it once for each of the 96 rows of $C$: $96 \times 576 = 55{,}296$ lines plus 1,152 for $A$ and $C$. Neither order reuses a line of $B$ before it is evicted; they differ only in the direction of the walk. At $n = 128$ a row of $B$ is 8 lines, the 128 lines of a column map to 8 of the 64 sets, and ijk, which walks a column, thrashes those sets (2,130,960 lines); ikj walks a row and does not (133,120). A third nest must reuse $B$ across several rows of $C$ before leaving it: blocking, which streams $B$ $n/T$ times.
2. Without the second barrier a fast invocation can begin the next step and overwrite the tile while a slow one is still reading it. With $K = T$ there is one step and nothing is overwritten, so the test passes; with $K = 64$ there are four steps and the race appears. (Chapter 5's missing-barrier test has the same shape.)
3. Tile $T = qr = 16$; registers $32 + 4 + 4 = 40$; shared memory $2 \times 16 \times 8 \times 4 = 1{,}024$ bytes; global loads $2n^3/T = 134{,}217{,}728$, the same as the tiled kernel with $T = 16$ (the loads depend on $T$ only); shared reads per multiply-add $2/r = 1$, half of the tiled kernel's 2. It has 64 invocations instead of 256 and holds 4 outputs each; its registers are 40 as in the tiled table.
4. $32 + 256 + 32 = 320$ registers per invocation, over the 255 an invocation can name, so it cannot run in the model; the block would have to be smaller, or the 256 invocations fewer with a smaller block, or the register file per invocation larger. Even with the registers, the halving is of global loads, which the global roof of the table shows is the limit only until the arithmetic peak is reached.
5. Word $a = 130 kk + r$ lies in bank $(2kk + r) \bmod 32$; the 32 lanes ($r = 0..3$, $kk = 0..7$) cover only 18 banks and the table computes a 2-way conflict, so it lowers the degree from 8 at stride 128 to 2, but not to 1. Stride 132 gives $4kk + r$, which is a bijection of the 32 lanes onto the 32 banks.
6. No: its intensity 56.89 is far below the ridge of 400, so the attainable rate is $56.89 \times 800 = 45{,}511$ (GFLOP/s), 0.142 of the matrix unit's 320,000. A faster unit moves the ridge up, so the same bytes per operation leave it further below its roof; it needs more reuse (or fewer bytes) to be fed.
7. Tiling does not change the order of the additions of an output when one accumulator is carried across the steps in ascending $k$, so the tiled kernel is bit-identical to the naive one (the table's first two rows agree in 16 of 16 outputs, and brain's `reg3_parity` test compares bit for bit). Split-K, a fused multiply-add and several accumulators do reorder. The test is correct for a kernel meant to keep the order, and a tolerance is needed only for one that does not.
8. Creating a pipeline succeeds on a driver that does not validate its capabilities, and says nothing about whether the device executes the fragment instructions correctly: the commit of the story reports exactly this on a device that enumerates no shapes. Only the query of the supported shapes and types is evidence, which is why the gate is the queried requirement.
