<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Project: A Device-Independent Tensor Engine

*Can you make the same matrix multiply run correctly and fast on more than one kind of hardware?*

A matrix product that is right on the machine where it was written has proved nothing about the next machine. Thirteen chapters have taken a matrix multiplication from three lines of arithmetic to a family of kernels, and have taught the habits that keep a family honest: count the work before timing it, run every kernel on a simulator that exposes its edges, derive a tolerance instead of choosing one, gate every candidate before it is allowed to be fast, and measure with a clock that has been waited for. This chapter asks you to turn those habits into one program. You will write, in a language of your choice, a small tensor engine with seven parts:

1. **A tensor and an oracle.** A strided tensor (shape, strides, offset over flat storage) and a reference matrix product in 64-bit arithmetic that every later result is judged against (Volume I, Chapter 5; Chapter 6).
2. **A CPU backend.** The backend interface of Chapter 9, a workgroup-and-barrier executor that runs kernels written as the program of one invocation (the simulator transformation of Chapters 5 and 9), and the naive matrix-product kernel on top of it.
3. **Tiled kernels.** The tiled and the register-blocked matrix products of Chapter 6 with shared memory and barriers, correct on sizes that are not multiples of the tile, with the memory traffic your executor must report as exact integers.
4. **Reductions and an epilogue.** Row sums and a global sum in two passes (Chapter 7), and the fused bias-and-GELU epilogue of Chapter 8 with its bytes before and after.
5. **Parity.** A second device configuration of the same executor, with another workgroup tile, another order of the additions and fused multiply-adds, and a parity test whose tolerance is derived from the error bound (Chapter 9), which also catches one kernel with a planted defect.
6. **A quantized product.** Group-wise int8 and int4 weights with the dequantization inside the matrix product (Chapter 12), and the error to expect.
7. **Selection and honest measurement.** Candidates that declare what they need, a reference gate, and a cost report in FLOPs, bytes and the fraction of a measured roofline from a seeded model clock with warm-up and repetition rules (Chapters 10 and 11), and the choice the engine must arrive at for stated shapes.

Four words recur and are defined by use. The **oracle** is the float64 reference product against which every device result is judged. A **device configuration** is the set of choices that make two devices run the same kernels differently: the workgroup tile, the order of the additions, whether a multiply followed by an add is fused. The **gate** is the check that a candidate passes before it may be timed. A **fixture** is a small input fixed by a printed generator seed, so that every language computes the identical numbers.

Nothing here is new. Every part was derived and worked by hand in Volume IV; the project asks whether the parts compose into an engine you would trust on a device you have not seen, which is a different thing from following a derivation. The failures it is designed to expose are the ordinary ones of this volume: a kernel that is right on tile-sized shapes and wrong on others, a barrier that a ragged edge skips, a tolerance chosen after looking at what passes, a fused kernel that changes the bits, a selector that picks the fastest kernel without asking whether it is right, and a timing that includes the transient of a cold device.

The project is built in order, and each milestone says what to build (in the book's pseudocode, which fixes the interface and nothing else), why it matters later, a check you run before moving on, and the mistakes readers actually make. The **acceptance tests** at the end give inputs and the outputs a correct engine produces, written so that any language can be tested against them. Every expected value was computed by a second, independent implementation, `scripts/book/worked/project_tensor_engine.py`, and is regenerated whenever the book is built. Use only your language's standard library. You need the book's generator (Volume I, Chapter 17), correctly rounded float32 arithmetic and a fused multiply-add (Volume IV, Chapters 1 and 9), SHA-256, and nothing else.

![Seven boxes labelled with the milestones. Milestone 1 (tensor and oracle) is at the top, with milestone 2 (CPU backend) below it and milestone 3 (tiled kernels) below that. Milestone 4 (reductions, epilogue) and milestone 5 (parity) are in the next row, 5 directly below 3 and 4 to its left. Milestone 6 (quantized product) is below 4 and milestone 7 (selection), highlighted, is at the bottom right; both receive an arrow from milestone 5.](img/engine-milestones.svg)

The figure shows which milestone feeds which, so the order 1 to 7 is a valid build order. The oracle of milestone 1 and the tolerance of milestone 5 are used by every check after them. Build and test each milestone before starting the next: a failure in milestone 7 that traces back to the executor of milestone 2 costs an afternoon, and the same failure caught by the check of milestone 2 costs a minute.

Before the first milestone, six questions bring back what it leans on. They reach across the volume, and the first reaches back to Volume I. Answer each from memory before reading on; the answers are at the end.

1. (Volume I, Chapter 5, [Arrays](../v01-mathematics/05-arrays.md)) A row-major array of shape $[3, 4]$ holds the numbers $0, 1, \ldots, 11$. What are the shape, the strides and the offset of its transpose as a view, and which storage position does element $[1, 2]$ of that view read?
2. (Chapter 5, [Writing Compute Kernels](05-compute-kernels.md)) A tiled kernel with $4 \times 4$ workgroups computes a $[13, 10]$ output and, to save a branch, executes `return` before its barrier for every invocation whose row or column is outside the output. How many of the workgroups of the grid reach the barrier with only some of their invocations?
3. (Chapter 6, [Making Matrix Multiplication Fast](06-matmul.md)) How many words does a tiled kernel with tile side $T = 4$ load from device memory for $C = AB$ with $m = 13$, $n = 10$, $k = 7$, when elements outside the matrices are not loaded? How many does the naive kernel load, and how many does the register-blocked kernel with $T = 8$ load?
4. (Chapter 9, [One Program, Many Devices](09-backends.md)) What is the bound on the error of a dot product of 100 products computed as a left fold, in units of $uS$? What is it for 128 products added as a tree of adjacent pairs, and why may a test hold the tree to the tighter bound?
5. (Chapter 11, [Measuring Performance Honestly](11-profiling.md)) A kernel is timed by reading the host clock right after `dispatch` returns. What did the clock measure, and what two additions make the number the cost of the kernel?
6. (Chapter 12, [Quantization](12-quantization.md)) A weight matrix of shape $[128, 24]$ is stored as 4-bit codes with a float32 scale per group of 32. How many bits per weight is that, and how many bytes does the matrix occupy, against float32?

## Milestones

The pseudocode follows [Pseudocode Conventions](../pseudocode.md): 0-indexed, row-major, explicit broadcasts, explicit randomness, `require` for every input assumption. Every `f32` operation rounds to float32 one operation at a time, as the pseudocode defines; a fused multiply-add is written `fma`. Store every matrix row-major. Every random input comes from the book's SplitMix64 at a seed printed beside the table, and the draws are taken in the order the table states. Write the checks of each milestone as a test file that you rerun after every change.

Results are compared by digest wherever a table would be too long to print. A digest is SHA-256 over the bytes of a result in row-major order, written as 64 lowercase hexadecimal digits and cut to the first 16, with each number as its IEEE 754 value in little-endian byte order: 4 bytes (binary32) for a result held in float32 and 8 bytes (binary64) for a result of the oracle. The check is exact: one different bit changes the digest. Counts are exact integers. A quantity that depends on a rounding in a library function (`exp`, `log`, `tanh`) is printed to the few digits that do not depend on the library, and the tolerance column says so.

### Milestone 1: the tensor and the oracle

The first thing to build is a tensor of two dimensions that is a view of flat storage, with two view operations that copy nothing, and the reference product.

```text
record Tensor:
    storage: array<f32>[S]
    shape: list<int>                        # [d0, d1]
    strides: list<int>                      # [s0, s1], in elements
    offset: int

function at(t: Tensor, i: int, j: int) -> f32:
    require 0 <= i < t.shape[0] and 0 <= j < t.shape[1]
    return t.storage[t.offset + i * t.strides[0] + j * t.strides[1]]

function transpose(t: Tensor) -> Tensor:        # the same storage, the shape and the strides exchanged
    return Tensor(t.storage, [t.shape[1], t.shape[0]], [t.strides[1], t.strides[0]], t.offset)

function row_slice(t: Tensor, lo: int, hi: int) -> Tensor:     # rows lo .. hi - 1: the offset moves
    require 0 <= lo < hi <= t.shape[0]
    return Tensor(t.storage, [hi - lo, t.shape[1]], t.strides, t.offset + lo * t.strides[0])

function matmul_oracle(a: Tensor, b: Tensor) -> array<real>[M, N]:
    # `real` is binary64 in this function: every product and every sum is a 64-bit operation
    require a.shape[1] == b.shape[0]
    c = zeros(real, [a.shape[0], b.shape[1]])
    for i in 0 .. a.shape[0]:
        for j in 0 .. b.shape[1]:
            acc = 0.0
            for p in 0 .. a.shape[1]:
                acc = acc + at(a, i, p) * at(b, p, j)      # a float32 times a float32 is exact in 64 bits
            c[i, j] = acc
    return c
```

The oracle is a left fold in float64. The product of two float32 numbers has at most 48 significant bits and is exact in float64, so the only rounding is that of the sums, with a bound of $k\,2^{-53}/(1 - k\,2^{-53})$ times $S$ for $k$ terms (the bound factor of Chapter 9 at the unit roundoff $2^{-53}$ of float64, written out in milestone 5), some $2^{29}$ times smaller than the float32 bounds the oracle will judge. The second function you need now is $S_{ij} = \sum_p |a_{ip}\, b_{pj}|$, the scale of every error bound in this chapter.

Every other milestone ends in a comparison with this function. A kernel that reads its operands through strides (milestone 2) is tested on views of the same storage, and the oracle reads the views in the same way, so a stride that is wrong in a kernel and right in the oracle cannot go unseen.

To check it, reproduce the four views of one storage, then a product through a transposed view:

<!-- worked: project_tensor_engine.tensor_views -->
Storage = [0, 1, ..., 11] (twelve float32 words), A = shape [3, 4] with the row-major strides (4, 1) and offset 0. A transpose is the same storage with the shape and the strides exchanged; a row slice moves the offset and keeps the strides. Element (i, j) of a view is storage[offset + i s0 + j s1]; the table reads the element at the index shown.

| view | shape | strides | offset | index | storage position | element | flat() equals element by element |
|---|---|---|---:|---|---:|---:|---|
| A | [3, 4] | [4, 1] | 0 | (1, 2) | 6 | 6 | yes |
| A transposed | [4, 3] | [1, 4] | 0 | (1, 2) | 9 | 9 | yes |
| A rows 1 to 2 | [2, 4] | [4, 1] | 4 | (1, 2) | 10 | 10 | yes |
| A rows 1 to 2, transposed | [4, 2] | [1, 4] | 4 | (0, 1) | 8 | 8 | yes |

A view in a product. seed 0xcd8ae5bab75fbc75 (14810902914664479861). Draw order: the 20 entries of X (shape [4, 5]) then the 15 entries of W (shape [3, 5], stored [n, k]), each `2 uniform() - 1` rounded to float32. The product X W^T taken through the transposed view of W has the digest 88c0c0ac2c8ad03b (binary64, row-major, 12 values); the same product with W^T copied out first has 88c0c0ac2c8ad03b; identical: yes.
<!-- /worked -->

Then the oracle, on integer matrices small enough that the product is exact in every format. The smallest case is printed in full; the others are checked by digest:

<!-- worked: project_tensor_engine.oracle_cases -->
Integer matrices for the oracle and the exact device tests. seed 0xd6cc7f538459bbcf (15477886016045038543). Draw order: the shapes (m, n, k) in the order of the table; for each, the m k entries of A row-major, then the k n entries of B row-major, each `integer(9) - 4` (so in -4 .. 4). The smallest case, m = 2, n = 2, k = 3: A = [[-2, 0, -2], [0, -4, 4]], B = [[-4, 3], [-4, 2], [-3, 2]], so C = A B = [[14, -10], [4, 0]]. 'FLOPs' is 2 m n k. Digests are SHA-256 over binary64 little-endian bytes, row-major, first 16 hex digits.

| m x n x k | FLOPs | digest of A | digest of B | digest of C (oracle) |
|---|---:|---|---|---|
| 2 x 2 x 3 | 24 | 6258cbc27e857aa2 | 9891fece18a462f7 | 7544addf15e64632 |
| 5 x 7 x 3 | 210 | 8b44d463bdb05759 | 44b99213de732fbd | 92c8db7ed81ed8cb |
| 8 x 8 x 8 | 1,024 | bb8db9f25371b53b | 3afbf099f5f867f7 | 21026af281237f0c |
| 13 x 11 x 9 | 2,574 | 1fe7442d13e0530a | c7a69cba59f26c3a | d3f0b318148f982f |
| 17 x 19 x 23 | 14,858 | 5fda9b1148c4a482 | b8fd4b7d1cc0d08a | 54e3b2fbfc449aec |
<!-- /worked -->

Three mistakes recur here. One is to store the transpose by copying it in the tests and by view in the engine, so that the stride code is never exercised; test through the view. Another is to take the oracle's accumulator in float32 "to match the kernel", when the oracle exists to be more accurate than the kernels it judges. The third is to subtract 4 from the draw `integer(9)` in unsigned arithmetic, which wraps: the entries are $-4, \ldots, 4$ and the subtraction must happen in a signed type.

### Milestone 2: the CPU backend and the naive kernel

The second milestone builds the seven operations of the backend interface of Chapter 9, a workgroup executor that runs a kernel written as the program of one invocation, and the naive matrix-product kernel.

```text
interface Backend:
    capabilities() -> record
    compile(source: text) -> kernel
    create_buffer(bytes: int) -> buffer
    upload(b: buffer, data)
    dispatch(k: kernel, buffers: list<buffer>, params: list<u32>, n_grp: int)
    read_back(b: buffer, count: int) -> array<f32>[count]
    release(b: buffer)
```

In this project a `kernel` is a value you build in your own language: its name, the names of the buffers it binds in order, its workgroup size, the number of shared words it needs for a given parameter block, and a function that returns the list of **phases** of one workgroup, the code between two consecutive barriers, each phase the program of one invocation. The parameter block is a record of named integers instead of a list of words; the contract is otherwise Chapter 9's. A grid is one integer, `n_grp`; a kernel with a two-dimensional grid of $n_x$ by $n_y$ workgroups takes $n_x$ in its parameters and computes $w_x = w \bmod n_x$, $w_y = \lfloor w / n_x \rfloor$.

The executor is the transformation of Chapter 9: a workgroup is a loop over its invocations, and a barrier is the end of a loop.

```text
function launch(mem: inout Memory, phases: list<function>, n_grp: int, wg: int, shared_words: int,
                params: record) -> (int, int):
    require n_grp >= 1 and wg >= 1
    blocks = (0, 0)                                  # transactions of 32 bytes: loads, stores
    for w in 0 .. n_grp:                             # workgroups in ascending order
        shared = zeros(f32, [shared_words])
        for t in 0 .. shared_words:
            shared[t] = 99.0                         # unspecified in the book; 99 makes a read of an unwritten slot visible
        alive = zeros(bool, [wg])
        for l in 0 .. wg:
            alive[l] = true
        for k in 0 .. len(phases):
            log = a list of wg empty lists           # the accesses of each invocation in this phase, in order
            for l in 0 .. wg:                        # the invocations of a workgroup, ascending
                if alive[l]:
                    run phases[k] for the invocation (w, l) over mem, shared, params,
                        appending (kind, address) to log[l] for every load and store;
                        a `return` in the phase sets alive[l] = false and ends only this phase
            blocks = blocks + transactions(log, wg)
            if k < len(phases) - 1:
                arrived = number of true entries of alive
                require arrived == 0 or arrived == wg, "a barrier that only some invocations reach"
    return blocks                                    # (load transactions, store transactions)
```

Three rules of this executor are the rules of Chapters 5 and 9, and each has a mistake that produces a plausible answer. A returned invocation stays returned in every later phase (the `alive` mask). The shared array is created afresh for every workgroup. A barrier that only some invocations reach is an error, not a result. The `transactions` function is the counting rule of Chapter 5: the invocations of a workgroup are cut into subgroups of 32 consecutive local indices; the $j$-th entry of the log of each invocation of a subgroup is one memory instruction (an invocation that returned has no entry, and one that skipped an access has its later entries moved up); the loads of one such instruction cost one transaction for each distinct 32-byte block that their addresses touch, and so do its stores, counted apart. Buffers are placed by the allocator of Chapter 5: each starts at a multiple of 8 words (32 bytes), words between buffers hold $-777$, and a buffer is filled with 1000 until it is uploaded or written.

Beside the transactions, your executor must report the **words** each kernel touched, per buffer: the number of single-word loads and stores. Count them in the memory, not in the kernel (a kernel that counts its own loads can forget one, or count one it did not issue); a load that lands outside every buffer of the kernel is counted under that name.

The naive kernel is Chapter 6's, with the strides of Chapter 5 so that it reads views:

```text
kernel matmul_naive(A: array<f32>[M, K], B: array<f32>[K, N], C: inout array<f32>[M, N],
                    M: int, N: int, K: int, oa: int, sa0: int, sa1: int, ob: int, sb0: int, sb1: int) workgroup n_wg:
    if g >= M * N: return
    i = g // N
    j = g mod N
    acc = 0.0
    for p in 0 .. K:
        acc = acc + A[oa + i * sa0 + p * sa1] * B[ob + p * sb0 + j * sb1]
    C[g] = acc

launch matmul_naive(...) over ceil(M * N / n_wg) workgroups
```

Here `A[...]` is a flat index into the buffer, and `oa`, `sa0`, `sa1` are the offset and strides of the tensor that the host uploaded (the whole storage, not a copy of the view). The host function `matmul(kind, a, b)` creates three buffers, uploads the storage of the two tensors, picks the grid, dispatches, reads $C$ back and releases the buffers. The workgroup size of the naive kernel is a device property (16 on the first device of milestone 5).

This kernel is the portable reference of the whole engine, the candidate that is always admitted and always present (Chapter 10). The executor under it is where every later defect becomes visible, because it is the only place where a load past the end of a buffer reads defined words.

To check it, run the naive kernel on the integer cases of milestone 1 (the same seed and draws), print the words and transactions, and compare $C$ with the oracle. On integers every sum is exact, so "equal" means equal:

<!-- worked: project_tensor_engine.backend_exact -->
The naive kernel of Chapter 6 (one output per invocation, workgroups of 16, the guard `g >= m n` returns) on the CPU backend, device A (left fold, no contraction), on the cases of `project_tensor_engine.oracle_cases` (same seed, same draws). Buffers are allocated in the order A, B, C, each starting at a multiple of 8 words (32 bytes) with -777 in the padding words; C starts as 1000. 'loads' and 'stores' count single words as the executor saw the kernel touch memory; 'blocks' are the 32-byte transactions of Chapter 5 (32-lane subgroups, one instruction per j-th access, one transaction per distinct block). 'Expected' is 2 m n k loads and m n stores.

| m x n x k | workgroups | invocations | loads of A | loads of B | stores of C | load blocks | store blocks | 2 m n k | C equals the oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 x 2 x 3 | 1 | 16 | 12 | 12 | 4 | 6 | 1 | 24 | yes |
| 5 x 7 x 3 | 3 | 48 | 105 | 105 | 35 | 25 | 5 | 210 | yes |
| 8 x 8 x 8 | 4 | 64 | 512 | 512 | 64 | 96 | 8 | 1,024 | yes |
| 13 x 11 x 9 | 9 | 144 | 1,287 | 1,287 | 143 | 369 | 18 | 2,574 | yes |
| 17 x 19 x 23 | 21 | 336 | 7,429 | 7,429 | 323 | 2313 | 41 | 14,858 | yes |
<!-- /worked -->

For the smallest case, with $A = \begin{pmatrix} -2 & 0 & -2 \\ 0 & -4 & 4 \end{pmatrix}$ and the $B$ printed in milestone 1, the kernel launches one workgroup of 16 invocations, of which 4 have an output, loads 12 words of $A$ and 12 of $B$, and stores 4. Any change to the guard `g >= M * N` shows in the loads at once.

Four mistakes cost readers time. Writing `return` for the guard but running the invocations of a barrier-free kernel as a single loop that `break`s makes the early invocations skip the rest of the grid. Counting a word in the kernel as it issues the load misses the load that an unguarded kernel makes outside the matrix, which milestone 5 needs. Giving each buffer its own memory region with a gap makes a read past the end of a buffer fault instead of reading the next buffer, and the layout above is part of the test. Packing the buffers without alignment changes the transaction counts and not the results.

### Milestone 3: tiled and register-blocked kernels

Third come the tiled matrix product and the register-blocked matrix product of Chapter 6, correct for every $m$, $n$, $k$.

The register-blocked kernel of Chapter 6 (`matmul_blocked`) is the model: a workgroup of $q \times q$ invocations computes a tile of side $T = q\,r$, each invocation an $r \times r$ block in registers, and the loop over $k$ moves in steps of $b_k$ with a barrier between loading the two tiles into shared memory and using them. The tiled kernel (`matmul_tiled`) is the same kernel with $r = 1$ and $b_k = T$, and one implementation serves both. Write it as a list of phases, a load phase and a compute phase for each step $c = 0, \ldots, \lceil k / b_k \rceil - 1$, the barrier being the boundary between phases. The kernel reads $A$ and $B$ through the strides of milestone 2 and keeps its accumulators in a local array that lives across phases (a per-invocation array indexed by the workgroup and local index, the promotion of a local across a barrier that Chapter 9 required). The tile loads use the index decomposition of Chapter 6's listing: the invocation copies the elements $idx = l,\ l + q^2,\ l + 2q^2, \ldots$ below $T b_k$ of each chunk, with $(\mathrm{row}, \mathrm{kk}) = (\lfloor idx / b_k \rfloor,\ idx \bmod b_k)$ for the chunk of $A$ and $(\mathrm{kk2}, \mathrm{col}) = (\lfloor idx / T \rfloor,\ idx \bmod T)$ for the chunk of $B$. The edges follow the guard rule of Chapter 5:

```text
load phase of step c, for the element idx of the A chunk and the B chunk that this invocation copies:
    a_tile[idx] = 0.0
    if w_y * T + row < M and c * b_k + kk < K:          # row, kk: the position of idx in the A chunk
        a_tile[idx] = A[oa + (w_y * T + row) * sa0 + (c * b_k + kk) * sa1]
    b_tile[idx] = 0.0
    if c * b_k + kk2 < K and w_x * T + col < N:         # kk2, col: the position of idx in the B chunk
        b_tile[idx] = B[ob + (c * b_k + kk2) * sb0 + (w_x * T + col) * sb1]
```

Elements outside the matrices are not loaded: the guards surround the loads, so the global loads of the kernel are exactly

$$
k\,\bigl(m \lceil n / T \rceil + n \lceil m / T \rceil\bigr)
$$

(every element of $A$ once per column tile, every element of $B$ once per row tile), and the naive kernel's are $2mnk$. The stores are $mn$ for every kernel. The last compute phase stores the block of the invocation where `row < M and col < N`.

These are the kernels that the parity test of milestone 5 compares across devices, that the epilogue of milestone 4 is fused into, and that the selection of milestone 7 chooses between. They are the only kernels in the engine whose loads depend on the tile, so the traffic numbers below are the engine's first measured gain, in words.

The check is two ragged products, on integers, with the three kernels at the same number of invocations per workgroup for the tiled and the blocked one (16): the tiled kernel with $T = 4$ ($q = 4$, $r = 1$), and the blocked kernel with $q = 4$, $r = 2$, $T = 8$, step depth 4. Your executor must report these exact integers:

<!-- worked: project_tensor_engine.tiled_traffic -->
Ragged products on the CPU backend, device A, integer matrices (`integer(9) - 4`; seed 0xeac58f1360da3e00 (16917084888679202304) for 13 x 10 x 7; seed 0x39ea5775c09ae0c0 (4173244167966351552) for 20 x 18 x 9; draw order: the m k entries of A, then the k n entries of B, row-major). The tiled kernel is the blocked kernel with r = 1 and step depth bk = T; the blocked kernel uses q = 4 invocations per side, r = 2, step depth 4. Elements outside the matrices are not loaded (the guards sit around the loads). Expected loads: naive 2 m n k; tiled and blocked k (m ceil(n / T) + n ceil(m / T)), T the side of the workgroup tile. Stores are m n for every kernel.

| m x n x k | kernel | workgroups | invocations | loads of A | loads of B | loads in all | formula | stores of C | load blocks | store blocks | C equals the oracle |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 13 x 10 x 7 | naive, 16 per workgroup | 9 | 144 | 910 | 910 | 1,820 | 1,820 | 130 | 248 | 17 | yes |
| 13 x 10 x 7 | tiled, T = 4 (16 invocations) | 12 | 192 | 273 | 280 | 553 | 553 | 130 | 191 | 45 | yes |
| 13 x 10 x 7 | blocked, q = 4, r = 2, T = 8 (16 invocations) | 4 | 64 | 182 | 140 | 322 | 322 | 130 | 100 | 70 | yes |
| 20 x 18 x 9 | naive, 16 per workgroup | 23 | 368 | 3,240 | 3,240 | 6,480 | 6,480 | 360 | 957 | 45 | yes |
| 20 x 18 x 9 | tiled, T = 4 (16 invocations) | 25 | 400 | 900 | 810 | 1,710 | 1,710 | 360 | 615 | 120 | yes |
| 20 x 18 x 9 | blocked, q = 4, r = 2, T = 8 (16 invocations) | 9 | 144 | 540 | 486 | 1,026 | 1,026 | 360 | 329 | 180 | yes |
<!-- /worked -->

The loads of the tiled kernel are 553 against the naive 1,820, and the blocked kernel's are 322: tiling cut the loads by the factor Chapter 6 derives, $2mnk$ over $k(m\lceil n/T\rceil + n\lceil m/T\rceil)$, and the larger tile cut them again. Notice what did not move: the stores, and the result. The second check is that tiling does not change the arithmetic. Every output of every kernel adds its terms in the order $p = 0, 1, \ldots, k - 1$, so on real-valued inputs the three kernels are bit-identical, not merely close:

<!-- worked: project_tensor_engine.summation_order_identity -->
Real-valued ragged product, m = 13, n = 10, k = 37 (k is not a multiple of any step depth used). seed 0xeb26f262a1d4d737 (16944497153463867191). Draw order: the 481 entries of A, then the 370 entries of B, row-major, each `2 uniform() - 1` rounded to float32. Device A: every multiply-add is two float32 operations (the product rounded, then the sum), each output a left fold over k. Digest: SHA-256 of the binary32 little-endian bytes of C, row-major, first 16 hex digits.

| kernel | digest of C | global loads |
|---|---:|---:|
| naive, 16 per workgroup | 5a0412aa6a31d727 | 9620 |
| tiled, T = 4 (16 invocations) | 5a0412aa6a31d727 | 2923 |
| blocked, q = 4, r = 2, T = 8 (16 invocations) | 5a0412aa6a31d727 | 1702 |

All three digests equal: yes. Tiling changes where the numbers are read from, not the order in which each output adds its terms. Against the float64 oracle: largest |C - oracle| / (u S) = 2.364, derived tolerance 37.0 u S (k_rd = 37); outside tolerance: 0.
<!-- /worked -->

The tolerance printed is the one milestone 5 derives; here it is only the statement that the left fold stays within $k$ roundings of the oracle.

The defects of this milestone show only on a ragged edge, so a test that uses only shapes that are multiples of the tile passes all of them. Loading out-of-range elements and multiplying them by a mask afterwards is one: the loads (and, for an out-of-range address, the memory read) happen anyway, and the counts above change. Dropping the barrier between the compute phase of step $c$ and the load phase of step $c + 1$ is another, which the phase list makes impossible by construction and a hand-written kernel makes easy; it passes every test with one step ($k \le b_k$) and fails with more. The last is accumulating in an order that depends on the tile ($r$ partial sums combined at the end): that is a legitimate kernel, and milestone 5 makes it a different device, but it is not this one, and the digests above will not match.

### Milestone 4: reductions and the fused epilogue

Fourth are the row sums and the global sum of Chapter 7, and the bias-and-GELU epilogue of Chapter 8, fused into the tiled kernel and as three separate kernels.

*Reductions.* The tree of Chapter 7 as phases: each invocation first adds its numbers into one value held in shared memory (a loop with the stride of the grid), then steps of stride $n_{\mathrm{wg}}/2, \ldots, 1$ add slot $l + s$ into slot $l$ for $l < s$, then invocation 0 stores. The row-sum kernel gives each row a workgroup:

```text
kernel reduce_rows(x: array<f32>[M, N], out: inout array<f32>[M], N: int) workgroup n_wg:
    shared s: array<f32>[n_wg]
    acc = 0.0
    j = l
    while j < N:
        acc = acc + x[w * N + j]
        j = j + n_wg
    s[l] = acc
    barrier
    stride = n_wg // 2
    while stride >= 1:
        if l < stride:
            s[l] = s[l] + s[l + stride]
        barrier
        stride = stride // 2
    if l == 0:
        out[w] = s[0]

launch reduce_rows(x, out, N) over M workgroups
```

The global sum is `reduce_pass` of Chapter 7 launched twice: pass 1 over $n_{\mathrm{grp}}$ workgroups leaves one partial each, pass 2 over one workgroup adds the partials (two dispatches, because nothing orders one workgroup against another inside a dispatch). Both fold their phases the way milestone 3 did: a loop that contains a barrier becomes a list of phases, one per stride.

*Epilogue.* Three kernels (`bias_add`, `gelu` and the product) against the product with the epilogue inside: after the accumulators are final, `u = acc + bias[col]` and the output is the GELU of $u$ with the nine operations of Chapter 8 in the order listed there, each rounded to float32, the constants $\sqrt{2/\pi}$ and $0.044715$ rounded to float32 first, and `tanh` the library function of the float32 argument rounded to float32.

A reduction and an epilogue are the two ways a matrix product is followed by something that is not a matrix product. The reduction is where the order of the additions is chosen by the kernel and not by the data, which is what lets milestone 5 derive a tolerance from the shape of the tree. The epilogue is the one fusion in the engine, and its bytes are the first number in the engine that measures the memory traffic a design avoids.

The check is row sums of an $[6, 37]$ array and a two-pass sum of 1000 numbers on each of the two devices of milestone 5 (reduction workgroups of 8 and of 16; five workgroups in pass 1 for both). The tolerance column is the derived one: the longest chain of roundings is the loop of an invocation (its first addition is exact), then the levels of the tree, and the global sum adds the chains of its two passes:

<!-- worked: project_tensor_engine.reductions -->
Row sums of an [6, 37] array X (one workgroup per row, each invocation adds the elements l, l + wg, ... of its row, then the tree of Chapter 7) and the two-pass sum of 1000 numbers Y (pass 1 over 5 workgroups with a grid-stride loop, pass 2 one workgroup over the 5 partials). seed 0xbc854a4019abb9b5 (13584345490202016181). Draw order: the 222 entries of X row-major, then the 1000 entries of Y, each `2 uniform() - 1` rounded to float32. Device A reduces with workgroups of 8, device B with workgroups of 16. 'k_rd' counts the roundings on the longest chain (an invocation's loop adds ceil(N / (n_grp wg)) numbers to a zero accumulator, the first addition being exact, then log2 wg levels of the tree; the global sum adds the chains of its two passes). 'Tolerance' is gamma(k_rd, u) in units of u S with u = 2^-24, S = sum |x|, plus the float64 oracle's own gamma(N - 1, 2^-53) S (a left fold in float64 is the oracle); 'error' is the largest |sum - oracle| / (u S). Digests of the row sums are SHA-256 over binary32 little-endian bytes (first 16 hex digits); a single total is printed as its 32 bits.

| reduction | wg | input | dispatches | loads | stores | digest or bits | k_rd | tolerance (u S) | error (u S) | outside tolerance |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| row sums, device A | 8 | 6 x 37 | 1 | 222 | 6 | 4f5634c503e94224 | 7 | 7.00 | 0.379 | 0 |
| global sum, device A | 8 | 1000 | 2 | 1005 | 6 | 0xc13b405b | 30 | 30.00 | 0.022 | 0 |
| row sums, device B | 16 | 6 x 37 | 1 | 222 | 6 | 02189cb01133f456 | 6 | 6.00 | 0.585 | 0 |
| global sum, device B | 16 | 1000 | 2 | 1005 | 6 | 0xc13b405a | 20 | 20.00 | 0.009 | 0 |

The two-pass sum of device A moves 1011 words (4,044 bytes) through memory for 1000 numbers: 4,000 bytes is the minimum, so the partials cost 1.1 percent more.
<!-- /worked -->

Both devices stay within their tolerance, by a wide margin, and the two global sums differ in the last bit of the float32 total. Neither is wrong; the tolerance is what lets the engine say so. The fused epilogue:

<!-- worked: project_tensor_engine.epilogue_bytes -->
The epilogue of Chapter 8, bias then GELU (the nine operations of Chapter 8 in the order listed there, every one rounded to float32, the constants sqrt(2 / pi) and 0.044715 rounded to float32, tanh the library tanh of the float32 argument rounded to float32) on the output of the tiled kernel (T = 4), m = 13, n = 10, k = 9. seed 0x62a92e2264a1c883 (7109264212033128579). Draw order: the 117 entries of A, the 90 entries of B (each `2 uniform() - 1` rounded to float32), then the 10 bias entries `0.5 normal()` rounded to float32. 'Words' is every single-word load and store the executor counted, over all dispatches; bytes are 4 times that; 'blocks' are 32-byte transactions. Digest: SHA-256 over binary32 little-endian bytes of the result, first 16 hex digits.

| design | dispatches | words through memory | bytes | transaction blocks | digest of the result |
|---|---:|---:|---:|---:|---|
| matmul, bias_add, gelu | 3 | 1491 | 5,964 | 392 | c0e871b6d001eba0 |
| matmul with the epilogue | 1 | 971 | 3,884 | 319 | c0e871b6d001eba0 |

Words saved: 520 = 4 m n = 520: the intermediate z and the intermediate u are each stored once and loaded once. The results are bit-identical: yes; both equal the direct evaluation of the formula: yes.
<!-- /worked -->

Fusion removed two intermediates, each stored once and loaded once, and nothing else; the result is bit-identical to the unfused chain because the fused kernel keeps the format of the intermediates and the order of the operations (Chapter 8).

A barrier inside a loop whose trip count differs between invocations (the grid-stride loop of invocation $l$ runs a different number of times from that of invocation $l + 1$) breaks the rule that every invocation must reach each barrier the same number of times; the loop ends before the first barrier and the tree that follows has its own, uniform loop. Padding the unused slots of a tree with a value that is not the identity of the sum is a second mistake. A third is fusing the epilogue with `fma(acc, 1.0, bias)` or contracting the GELU's multiply-add: the unfused kernels round the product, and the digest above will differ. The last is counting the bias as one load per output in one design and one load per column in the other; the executor counts words, so the two designs load the bias the same number of times.

### Milestone 5: parity across two devices

Fifth, a second device configuration of the executor, and a parity test.

```text
record DeviceConfig:
    tile: int                # side of the tiled kernel's workgroup tile
    lanes: int               # accumulators over which a dot product's additions are dealt
    contract: bool           # a product followed by an addition is one fused multiply-add
    reduce_wg: int           # workgroup size of the reductions
    naive_wg: int            # workgroup size of the naive kernel

function mad(contract: bool, acc: f32, a: f32, b: f32) -> f32:
    if contract:
        return fma(a, b, acc)
    return acc + a * b        # the product is rounded, then the sum
```

Device A is the device of milestones 2 to 4: tile 4, one accumulator, no contraction, reduction workgroups of 8, naive workgroups of 16. Device B differs in all five: tile 8, four accumulators, contraction, reduction workgroups of 16, naive workgroups of 32. The accumulator rule is: term $t$ of a dot product (the global index $t$ along $k$, not the position in a tile) goes to accumulator $t \bmod \mathrm{lanes}$, each accumulator is a left fold, and the accumulators are added left to right at the end. With one accumulator this is the left fold of device A, and with four it is what a vector unit does. The longest chain of roundings of a dot product of $k$ terms is

$$
k_{\mathrm{rd}} = \lceil k / \mathrm{lanes} \rceil + \mathrm{lanes} - 1 .
$$

The parity test is the procedure of Chapter 9 with the tolerance computed, never tuned:

```text
function gamma(j: int, u: real) -> real:
    require j * u < 1
    return j * u / (1 - j * u)

function parity_check(out: array<f32>[M, N], ref: array<real>[M, N], s: array<real>[M, N], k_rd: int, k: int)
        -> (int, real):
    # returns the number of outputs outside tolerance and the largest error over the tolerance of its own entry
    outside = 0
    worst = 0.0
    for i in 0 .. M:
        for j in 0 .. N:
            tol = (gamma(k_rd, 2^-24) + gamma(k, 2^-53)) * s[i, j]      # the device's bound plus the oracle's own
            err = abs(out[i, j] - ref[i, j])
            if not is_finite(out[i, j]) or err > tol:
                outside = outside + 1
            worst = max(worst, err / tol)
    return (outside, worst)
```

The tolerance of two devices against each other is the sum of their two float32 bound factors times $S$, $\bigl(\mathrm{gamma}(k_{\mathrm{rd},A}, 2^{-24}) + \mathrm{gamma}(k_{\mathrm{rd},B}, 2^{-24})\bigr)\,S$: two programs, each within its own bound of the same exact value, differ by at most the sum of the bounds. The criterion is the one that Chapter 9 assigns each kernel: where the kernel fixes every rounding and no device may contract, bit-identity; otherwise a tolerance derived from $k_{\mathrm{rd}}$ and $S$. The transposition of a tensor and the four-slot sum are bit-identical on every device, and a tolerance would only hide a defect there.

The parity test is also the gate of milestone 7: a candidate is admitted when this function reports nothing outside the tolerance on the shapes the gate covers. A test that cannot fail is not evidence, so the milestone ends with a kernel it must reject.

The check is both devices on three seeded real-valued products, the naive and the tiled kernel on each. The kernels of one device are bit-identical to each other (the digests of the naive and tiled rows agree on every shape), the two devices are not, and every error is far inside its bound:

<!-- worked: project_tensor_engine.parity_devices -->
Two device configurations of the CPU backend (the same kernels, different compilers).

| device | tile of the tiled kernel | naive workgroup | accumulators (lanes) | multiply-add | reduction workgroup |
|---|---|---|---|---|---:|
| A | 4 x 4 (16 invocations) | 16 | 1 | product, then sum | 8 |
| B | 8 x 8 (64 invocations) | 32 | 4 | fused multiply-add | 16 |

The lane rule: term t of a dot product goes to accumulator t mod lanes, each accumulator is a left fold, and the accumulators are added left to right at the end; the longest chain has k_rd = ceil(k / lanes) + lanes - 1 roundings. The tolerance of a device against the float64 oracle (a left fold in float64) is (gamma(k_rd, 2^-24) + gamma(k, 2^-53)) S per output, gamma(j, u) = j u / (1 - j u), S = sum |a b|; the tolerance of two devices against each other is (gamma(k_rd,A, 2^-24) + gamma(k_rd,B, 2^-24)) S. Errors are in units of u S (u = 2^-24); 'worst ratio' is the largest error over the tolerance of its own entry. seed 0x676b2cedcafff9e9 (7452099408239262185) for 13 x 10 x 7; seed 0x07876112043e445c (542509010129929308) for 21 x 18 x 100; seed 0xe698dc8d83a57bbd (16616273325540604861) for 33 x 17 x 200. Draw order: the m k entries of A, then the k n entries of B, row-major, each `2 uniform() - 1` rounded to float32. Digests: binary32 little-endian, first 16 hex digits.

| m x n x k | kernel | device | k_rd | tolerance (u S) | largest error (u S) | worst ratio | outside tolerance | digest or agreement |
|---|---|---|---:|---:|---:|---:|---:|---|
| 13 x 10 x 7 | naive | A | 7 | 7.00 | 1.796 | 0.2566 | 0 | 8f2fecc5f15904a8 |
| 13 x 10 x 7 | naive | B | 5 | 5.00 | 1.936 | 0.3873 | 0 | 5d9c41fde2819477 |
| 13 x 10 x 7 | naive | A vs B | 7 + 5 | 12.00 | 1.701 | 0.1418 | 0 | 54 of 130 equal |
| 13 x 10 x 7 | tiled | A | 7 | 7.00 | 1.796 | 0.2566 | 0 | 8f2fecc5f15904a8 |
| 13 x 10 x 7 | tiled | B | 5 | 5.00 | 1.936 | 0.3873 | 0 | 5d9c41fde2819477 |
| 13 x 10 x 7 | tiled | A vs B | 7 + 5 | 12.00 | 1.701 | 0.1418 | 0 | 54 of 130 equal |
| 21 x 18 x 100 | naive | A | 100 | 100.00 | 1.919 | 0.0192 | 0 | 1383dd854584b887 |
| 21 x 18 x 100 | naive | B | 28 | 28.00 | 0.665 | 0.0237 | 0 | 0e7a890eed479b06 |
| 21 x 18 x 100 | naive | A vs B | 100 + 28 | 128.00 | 1.882 | 0.0147 | 0 | 51 of 378 equal |
| 21 x 18 x 100 | tiled | A | 100 | 100.00 | 1.919 | 0.0192 | 0 | 1383dd854584b887 |
| 21 x 18 x 100 | tiled | B | 28 | 28.00 | 0.665 | 0.0237 | 0 | 0e7a890eed479b06 |
| 21 x 18 x 100 | tiled | A vs B | 100 + 28 | 128.00 | 1.882 | 0.0147 | 0 | 51 of 378 equal |
| 33 x 17 x 200 | naive | A | 200 | 200.00 | 1.740 | 0.0087 | 0 | cd374be0c40215dd |
| 33 x 17 x 200 | naive | B | 53 | 53.00 | 0.842 | 0.0159 | 0 | 33317150186b2a0c |
| 33 x 17 x 200 | naive | A vs B | 200 + 53 | 253.00 | 1.739 | 0.0069 | 0 | 59 of 561 equal |
| 33 x 17 x 200 | tiled | A | 200 | 200.00 | 1.740 | 0.0087 | 0 | cd374be0c40215dd |
| 33 x 17 x 200 | tiled | B | 53 | 53.00 | 0.842 | 0.0159 | 0 | 33317150186b2a0c |
| 33 x 17 x 200 | tiled | A vs B | 200 + 53 | 253.00 | 1.739 | 0.0069 | 0 | 59 of 561 equal |
<!-- /worked -->

The errors stay under $2\,uS$ at every length, against tolerances of 5 to 253 $uS$: the bounds are worst cases, reached only if every rounding errs the same way, and what matters for a test is that a correct device can never exceed them (Chapter 9). Device B's smaller $k_{\mathrm{rd}}$ is the effect of the four accumulators, and the pair tolerance, the sum of the two, is the one that applies when neither device is the reference.

*The planted defect.* The tiled kernel with the bound check of the $k$ edge removed from its two tile loads (`ka < K` and `kb < K`) is correct whenever $k$ is a multiple of the step depth, because then no load goes past the edge, and wrong otherwise: the loads read the words after the row, or after the buffer, and add products of garbage to every output. Run your parity test over five shapes:

<!-- worked: project_tensor_engine.broken_kernel -->
The tiled kernel (T = 4, step depth 4) with and without the bound check of the k edge on its two tile loads (`ka < K` and `kb < K` removed; the checks on rows and columns remain), tested on device A against the float64 oracle with the derived tolerance (k_rd = k). Real-valued inputs, `2 uniform() - 1` rounded to float32, the m k entries of A then the k n entries of B, row-major, from the generator of the name `project_tensor_engine.broken.<m>x<n>x<k>` seed 0x8b66696b041b270f (10044831927247775503) for 8 x 8 x 8; seed 0x229b6d129b220749 (2493706745359238985) for 12 x 8 x 16; seed 0x441f973fbb69b97f (4908808418840131967) for 13 x 10 x 7; seed 0x35c2a8073028b797 (3873843378318260119) for 9 x 6 x 5; seed 0xeb6c47148eda023a (16964012050019385914) for 16 x 16 x 20. Buffers A, B, C are allocated in this order at 32-byte boundaries with -777 in the padding and 1000 in C before the launch, and the workgroups run in ascending order, so that the words an unguarded load reads past the end of its buffer are defined. 'Outside' counts outputs beyond the tolerance or not finite, out of m n; 'worst ratio' is the largest error over the tolerance (printed as 'above 1e6' when it is: its value depends on the words the load happened to read).

| m x n x k | k a multiple of 4 | guarded: outside | guarded: worst ratio | guarded: verdict | unguarded: outside | unguarded: worst ratio | unguarded: verdict |
|---|---|---:|---:|---|---:|---:|---|
| 8 x 8 x 8 | yes | 0 of 64 | 0.188 | PASS | 0 of 64 | 0.188 | PASS |
| 12 x 8 x 16 | yes | 0 of 96 | 0.123 | PASS | 0 of 96 | 0.123 | PASS |
| 13 x 10 x 7 | no | 0 of 130 | 0.302 | PASS | 130 of 130 | above 1e6 | FAIL |
| 9 x 6 x 5 | no | 0 of 54 | 0.245 | PASS | 54 of 54 | above 1e6 | FAIL |
| 16 x 16 x 20 | yes | 0 of 256 | 0.122 | PASS | 0 of 256 | 0.122 | PASS |
<!-- /worked -->

The kernel passes at the three shapes whose $k$ is a multiple of 4, which is every shape a test would pick by habit, and fails at the two that are not, on every output. Your test must report the failure at $13 \times 10 \times 7$ with all 130 outputs outside the tolerance, and must pass the guarded kernel on all five shapes. The size of the worst ratio at the failing shapes depends on the words that happen to lie after the buffer, which is why the table prints only that it is above $10^6$; the count of outputs outside the tolerance does not.

Comparing the two devices with the larger of their two tolerances, or with the oracle's, is the first mistake: the first accepts a defect of one device that the other's bound hides, and the second is the wrong bound. Choosing the tolerance by running the test and picking a number above the largest error is another; the derived value has a reason, and an observed maximum is a fact about the seeds. Holding device B to the bound of device A's order, $k_{\mathrm{rd}} = k$, passes and leaves the test with no teeth for a defect in the lanes. Computing $S$ in float32 is a fourth, since $S$ is the scale of the bound and should be exact to its first digits; compute it in float64. The last is running the parity test on the shapes where the defect hides: the shape list is part of the test, and it must contain a ragged edge in every dimension.

### Milestone 6: group-wise quantized weights

Sixth, the quantizer of Chapter 12 for symmetric codes with a float32 scale, and a matrix product that decodes inside the loop.

```text
function quantize_groups(w: array<f32>[n], G: int, nbits: int) -> (scales: array<f32>[n // G], q: array<int>[n]):
    require n mod G == 0, "a group would straddle the end of the tensor"
    require 2 <= nbits <= 8, "unsupported width"
    qmax = 2^(nbits - 1) - 1
    scales = zeros(f32, [n // G])
    q = zeros(int, [n])
    for g in 0 .. n // G:
        amax = max([abs(w[g * G + t]) for t in 0 .. G])
        if amax > 0.0:
            scales[g] = amax / qmax                 # rounded to float32: the codes use the stored scale
            for t in 0 .. G:
                q[g * G + t] = clamp(round_even(w[g * G + t] / scales[g]), -qmax, qmax)     # the quotient in 64 bits
    return (scales, q)
```

This is the function of Chapter 12 with the scale stored in float32 (the format that the engine's weight tables use: $4 + 32/32 = 5$ bits per weight at 4 bits and a group of 32). The weight matrix $W$ of shape $[k, n]$ is quantized along $k$ for each column: the codes of column $j$ are `Q[j * K + t]` and its scales `S[j * (K / G) + g]`, so that one output reads two contiguous runs. The product keeps the scale out of the inner loop, the factored form of Chapter 12:

```text
kernel matmul_q(A: array<f32>[M, K], Q: array<int>[N * K], S: array<f32>[N * (K / G)], C: inout array<f32>[M, N],
                M: int, N: int, K: int, G: int) workgroup n_wg:
    if g >= M * N: return
    i = g // N
    j = g mod N
    total = 0.0
    for grp in 0 .. K // G:
        scale = S[j * (K // G) + grp]
        inner = 0.0
        for t in 0 .. G:
            inner = inner + A[i * K + grp * G + t] * Q[j * K + grp * G + t]      # the code is an exact small integer
        total = total + scale * inner
    C[g] = total
```

A second kernel decodes each weight first (`inner = inner + A[...] * (scale * Q[...])`) and is the comparison. The simulator keeps one code in a word; the stored size of the matrix is computed from the format, $n_b + 32/G$ bits per weight, because that is what a device that packs the codes reads.

This is the kernel that a model with quantized weights runs on every layer. It also shows the tolerance of milestone 5 on a kernel whose longest chain is not the length of the dot product, and separates the two errors the quantized product has: the error of the weights, which is the point of the format and is large, and the error of the arithmetic, which must be small.

The check is a product $[6, 128] \times [128, 24]$, with Gaussian weights and activations, both widths, both forms. Each row reports the quantization error (the kernel against the float64 product of the *original* weights, the number to expect) and the arithmetic error (the kernel against the float64 product of the *decoded* weights, which must stay under its derived tolerance):

<!-- worked: project_tensor_engine.quantized_matmul -->
The product C = A W, A of shape [6, 128] (activations), W of shape [128, 24] (weights), W quantized along k in groups of G = 32 for each column (symmetric codes -qmax .. qmax, qmax = 2^(bits - 1) - 1, scale = amax / qmax stored as float32, code = clamp(round_even(w / scale), -qmax, qmax)). seed 0x546969a482b88101 (6082508927029248257). Draw order: the 3072 weights `0.05 normal()` rounded to float32, row-major over [k, n], then the 768 activations `normal()` rounded to float32, row-major. Device A. 'Bits per weight' counts the 32-bit scale of every group: bits + 32 / G. 'Weight SQNR' is 10 log10 (mean w^2 / MSE of the decoded weights); 'output SQNR' and 'relative error' compare the kernel's C with the float64 product of the original weights (the error to expect from quantizing); 'worst ratio' compares the kernel's C with the float64 product of the decoded weights scale x code (the float32 arithmetic alone) over a tolerance (gamma(k_rd, 2^-24) + gamma(k, 2^-53)) S with S = sum |a w'|, k_rd = G + k / G for the factored form (the group's chain, the scale's product, the additions across groups) and k + 1 for dequantizing first (the product with the rounded weight scale x code adds one rounding to the chain of k). Digest: binary32 little-endian bytes of C, first 16 hex digits.

| weights | form | bits per weight | bytes per weight | weight SQNR (dB) | output SQNR (dB) | relative error (%) | k_rd | worst ratio | outside | digest |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| int8 | factored | 9 | 1.1250 | 45.31 | 45.02 | 0.561 | 36 | 0.0339 | 0 | 36be81209dd2f407 |
| int8 | dequantize first | 9 | 1.1250 | 45.31 | 45.02 | 0.561 | 129 | 0.0064 | 0 | b0e1bd9e3aac125f |
| int4 | factored | 5 | 0.6250 | 20.15 | 20.22 | 9.749 | 36 | 0.0256 | 0 | d093485656da7628 |
| int4 | dequantize first | 5 | 0.6250 | 20.15 | 20.22 | 9.749 | 129 | 0.0091 | 0 | 8b58548a0abed7b1 |

What the factored kernel reads (executor words) and the stored size of the weights:

| weights | loads of A | loads of Q | loads of S | stores of C | bytes of W in float32 | bytes of W stored |
|---|---:|---:|---:|---:|---:|---:|
| int8 | 18,432 | 18,432 | 576 | 144 | 12,288 | 3,456 |
| int4 | 18,432 | 18,432 | 576 | 144 | 12,288 | 1,920 |
<!-- /worked -->

The weight SQNR and the output SQNR agree to within a few tenths of a decibel, which is the prediction of Chapter 12 for white inputs: 20.15 dB of the weights at 4 bits becomes 20.22 dB at the output, a relative error of 9.7 percent, and 8 bits gives 45 dB and 0.56 percent. The two forms have the same quantization error, since they compute the same decoded weights; they differ in the last bits, within the tolerance each one's chain gives it (36 roundings for the factored form, 129 for decoding first). The stored size is the gain: 5 bits per weight is 1,920 bytes where float32 needs 12,288, the saving of 6.4 times in the bytes that a bandwidth-bound product reads.

Taking the scale from the float64 maximum and then using the unrounded value in the decoder is a trap: the codes must be computed with the stored (float32) scale, or the decoded weights are not the weights the kernel uses. Quantizing along $n$ instead of along $k$ lets a group straddle outputs, and the factored form no longer applies. Rounding $w / \mathrm{scale}$ half away from zero differs from the rule, round to nearest even, on exact ties, which a coarse scale produces. And comparing the kernel with the original weights and calling a 9.7 percent error a bug, or with the decoded weights and calling a 45 dB error a feature, mixes up two comparisons that answer different questions.

### Milestone 7: candidates, a gate and an honest choice

The last milestone puts the selection pipeline of Chapter 10 on the engine, with the measurement protocol of Chapter 11.

*Candidates and capabilities.* Four candidates of the matrix product: the naive kernel, `tiled` (the blocked kernel with $q = 8$, $r = 4$, step depth 8, so $T = 32$), `reg` ($q = 16$, $r = 8$, step depth 8, $T = 128$), and `reg-nok`, which is `reg` with the bound check of the $k$ edge removed, the defect of milestone 5 on a faster-looking kernel. A candidate declares what it needs, and you read the declaration from its kernel: barriers, bytes of shared memory ($2\,T\,b_k$ words of 4 bytes), the workgroup size ($q^2$). A device is the `capabilities()` record of Chapter 9, and the filter reads the record and never the name of the device:

```text
function refusal(record: record, c: Candidate) -> optional text:
    if c.barriers > record.max_barriers: return "no workgroup barriers"
    if c.shared_bytes > record.shared_bytes: return "workgroup memory"
    if c.workgroup > record.max_workgroup: return "workgroup size"
    return none
```

<!-- worked: project_tensor_engine.selection_capabilities -->
The two model units of Chapter 10 (`kernel_selection.DEVICES`) as the `capabilities()` record of Chapter 9; the CPU unit's compiler cannot run a barrier kernel (Chapter 9), so its record says 0 barriers.

| unit | max_workgroup | shared_bytes | subgroup | max_barriers |
|---|---:|---:|---:|---:|
| CPU unit | 1 | 0 | 0 | 0 |
| GPU unit | 256 | 16,384 | 32 | unbounded |

The catalogue. A candidate's declaration is read from its kernel: barriers (yes if it has any), bytes of shared memory (2 T bk words of 4 bytes), and its workgroup size (q x q invocations). 'tiled' is the blocked kernel with q = 8, r = 4, bk = 8 (T = 32), 'reg' with q = 16, r = 8, bk = 8 (T = 128), 'naive' one output per invocation; 'reg-nok' is 'reg' with the bound check of the k edge removed from its two tile loads (the defect of milestone 5). The filter reads the record, never the name of the device.

| candidate | barriers | shared bytes | workgroup | CPU unit | GPU unit |
|---|---:|---:|---:|---|---|
| naive | no | 0 | any | runs | runs |
| tiled | yes | 2,048 | 64 | refused: no workgroup barriers | runs |
| reg | yes | 8,192 | 256 | refused: no workgroup barriers | runs |
| reg-nok | yes | 8,192 | 256 | refused: no workgroup barriers | runs |

On the CPU unit the filter leaves only the naive kernel: with one candidate there is nothing to choose between, so the engine runs it without a gate or a measurement.
<!-- /worked -->

*The gate.* Run each surviving candidate's kernel through milestone 5's test on a shape that is a whole number of its tile and a shape that is ragged in every dimension, $(T, 16, 16)$ and $(T + 1, 17, 19)$ for tile side $T$ (8 for the naive kernel), on device A, with seeded real inputs. A candidate is admitted only if every output of both shapes is within the derived tolerance and finite.

<!-- worked: project_tensor_engine.selection_gate -->
The reference gate of Chapter 10 run on the engine: each candidate's kernel on the whole-tile shape (T, 16, 16) and the ragged shape (T + 1, 17, 19) of its tile T (T = 8 for the naive kernel, 32 for tiled, 128 for reg and reg-nok), device configuration A with one accumulator and no contraction (k_rd = k), real-valued inputs `2 uniform() - 1` rounded to float32 (the m k entries of A, then the k n entries of B, row-major; seed 0x5d9364b732bb7e58 (6742843805096705624) for 8 x 16 x 16; seed 0x207f511da57240c6 (2341679519027577030) for 9 x 17 x 19; seed 0x0767542706e0af86 (533487607462539142) for 32 x 16 x 16; seed 0x74a49a87c6224c38 (8405012712515587128) for 33 x 17 x 19; seed 0xa757f2ee22d66f06 (12058373631908409094) for 128 x 16 x 16; seed 0x46c7ea78364bb8fb (5100302905048283387) for 129 x 17 x 19). The same seeded inputs are used by every candidate of a shape. A candidate is admitted when every output of both shapes is within (gamma(k, 2^-24) + gamma(k, 2^-53)) S of the float64 oracle and finite.

| candidate | shape | outside tolerance | worst ratio | verdict |
|---|---|---:|---:|---|
| naive | 8 x 16 x 16 | 0 of 128 | 0.0719 | admitted |
| naive | 9 x 17 x 19 | 0 of 153 | 0.0771 | admitted |
| naive | gate verdict |  |  | ADMITTED |
| tiled | 32 x 16 x 16 | 0 of 512 | 0.1446 | admitted |
| tiled | 33 x 17 x 19 | 0 of 561 | 0.0953 | admitted |
| tiled | gate verdict |  |  | ADMITTED |
| reg | 128 x 16 x 16 | 0 of 2048 | 0.1588 | admitted |
| reg | 129 x 17 x 19 | 0 of 2193 | 0.1320 | admitted |
| reg | gate verdict |  |  | ADMITTED |
| reg-nok | 128 x 16 x 16 | 0 of 2048 | 0.1588 | admitted |
| reg-nok | 129 x 17 x 19 | 2193 of 2193 | above 1e6 | REFUSED |
| reg-nok | gate verdict |  |  | REFUSED |
<!-- /worked -->

`reg-nok` passes the whole-tile shape with the exact errors of `reg` and fails the ragged one on every output. A gate that ran only whole-tile shapes would have admitted it, and the cost report below shows what that would have bought.

*The cost report.* The unit is the GPU unit of Chapter 10 with its measured roofline, $F_{\mathrm{ms}} = 34{,}469$ (GFLOP/s) and $\mathrm{BW}_{\mathrm{ms}} = 646$ (GB/s) (`profiling.F_MS` and `profiling.BW_MS`). A candidate's cost on a shape is the cost model of Chapter 10 (`kernel_selection`: the equivalent bytes of a launch that pays its tile padding, its occupancy and one launch), divided by the bandwidth to give the time that the model clock then perturbs: each run is the noise-free time times $(1 + 0.04\,u_0)\,(1 - 0.25 \ln(1 - u_2)$ if $u_1 < 0.2$, else $1)$ and the warm-up transient $1 + 0.8\,e^{-i/5}$, with 20 noise-free times added to run 0. `reg-nok` is priced at 0.95 of `reg`, the saving the model grants for the checks it dropped. **This clock is a model**: every time below is its output, labelled as such, and the quantities that carry over to a real device are the work (FLOPs and bytes) and the fraction. The protocol is Chapter 11's: 300 dispatches per candidate and shape timed by the device's own stamps, the warm-up ended by `warmup_end` with $k = 5$ and tolerance $0.05$ and discarded, at least $264 = 4 \times 66$ steady runs (the $n_4$ of Chapter 11 for a 5 percent gap), the mean with its standard error, and the median for the fraction. The choice is the admitted candidate with the lowest mean, and it is accepted when the runner-up is slower by more than two standard errors of the difference.

```text
function choose(live: list<Candidate>, gate: function(Candidate) -> bool, steady: function(Candidate) -> array<real>[S])
        -> Candidate:
    require len(live) >= 1
    if len(live) == 1:
        return live[0]                   # nothing to choose between: no gate, no measurement
    admitted = empty list
    for c in live:
        if gate(c):
            append c to admitted
    require len(admitted) >= 1, "the reference is always admitted"
    best = admitted[0]
    for c in admitted[1:]:
        if mean(steady(c)) < mean(steady(best)):
            best = c
    return best
```

The report also measures the gate's refusals (an engine would not; the table does, to show what measurement alone would have chosen), so the draws of `reg-nok` are in the stream. On the CPU unit of Chapter 10 the filter leaves one candidate and the engine returns it unmeasured.

<!-- worked: project_tensor_engine.selection_report -->
The GPU unit of Chapter 10 with its measured roofline, 34,469 (GFLOP/s) and 646 (GB/s) (`profiling.F_MS`, `profiling.BW_MS`), and the model clock of Chapters 10 and 11: a run of a candidate is its noise-free cost (the cost in equivalent bytes of Chapter 10 over the bandwidth of the unit, times 0.95 for reg-nok, which the model grants for the checks it dropped) multiplied by the noise factor (1 + 0.04 u0)(1 - 0.25 ln(1 - u2) if u1 < 0.2, else 1) and by the warm-up transient 1 + 0.8 exp(-i / 5), plus 20 noise-free costs on run 0. Everything below is the output of this model, not a measurement of a machine. Protocol: 300 dispatches per candidate timed by the device's own stamps; the warm-up ends at the first run that starts 5 consecutive runs each within 5 percent of the minimum of runs 150 to 299, and those runs are discarded; the rest (at least 264 = 4 x 66, the runs for a 5 percent gap at four standard errors, Chapter 11) give the mean, its standard error s / sqrt(n) and the median; the winner is the admitted candidate with the lowest mean. seed 0x0711b23c5c9e3d12 (509384205191232786). Draw order: requests in the order of the table, candidates in catalogue order (those the unit can run), runs 0 .. 299, three uniforms each (u0, u1, u2). 'FLOPs executed' and 'bytes requested' are the counts of Chapter 10 for the candidate's tile (padded tiles included). Times are in microseconds of the model clock; the fraction is the roofline time of the problem on the measured roofline over the median (a fraction near 1 is the roofline).

| m x n x k | candidate | FLOPs executed | bytes requested | warm-up ends at run | steady runs | mean | standard error | median | fraction of roofline | status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 x 2560 x 2048 | naive | 10,485,760 | 41,953,280 | 19 | 281 | 439.72 | 2.87 | 424.79 | 0.076 | admitted |
| 1 x 2560 x 2048 | tiled | 335,544,320 | 21,637,120 | 16 | 284 | 39.88 | 0.28 | 38.56 | 0.843 | chosen |
| 1 x 2560 x 2048 | reg | 1,342,177,280 | 21,145,600 | 22 | 278 | 109.03 | 0.90 | 103.23 | 0.315 | admitted |
| 1 x 2560 x 2048 | reg-nok | 1,342,177,280 | 21,145,600 | 27 | 273 | 102.34 | 0.85 | 97.99 | 0.332 | REFUSED by the gate |
| 48 x 2560 x 2048 | naive | 503,316,480 | 2,013,757,440 | 21 | 279 | 3279.43 | 30.12 | 3113.16 | 0.011 | admitted |
| 48 x 2560 x 2048 | tiled | 671,088,640 | 73,891,840 | 18 | 282 | 124.77 | 1.07 | 119.06 | 0.284 | admitted |
| 48 x 2560 x 2048 | reg | 1,342,177,280 | 29,327,360 | 32 | 268 | 108.43 | 1.03 | 103.01 | 0.328 | chosen |
| 48 x 2560 x 2048 | reg-nok | 1,342,177,280 | 29,327,360 | 27 | 273 | 104.08 | 1.37 | 98.07 | 0.345 | REFUSED by the gate |
| 200 x 2560 x 2050 | naive | 2,099,200,000 | 8,398,848,000 | 17 | 283 | 13436.51 | 82.13 | 12990.94 | 0.005 | admitted |
| 200 x 2560 x 2050 | tiled | 2,351,104,000 | 280,192,000 | 16 | 284 | 454.32 | 3.32 | 438.17 | 0.139 | admitted |
| 200 x 2560 x 2050 | reg | 2,686,976,000 | 76,832,000 | 21 | 279 | 129.66 | 1.19 | 123.70 | 0.492 | chosen |
| 200 x 2560 x 2050 | reg-nok | 2,686,976,000 | 76,832,000 | 20 | 280 | 122.95 | 0.98 | 117.58 | 0.518 | REFUSED by the gate |
| 512 x 2560 x 2048 | naive | 5,368,709,120 | 21,480,079,360 | 24 | 276 | 34559.77 | 284.59 | 33182.66 | 0.005 | admitted |
| 512 x 2560 x 2048 | tiled | 5,368,709,120 | 676,331,520 | 23 | 277 | 1104.85 | 10.65 | 1052.44 | 0.148 | admitted |
| 512 x 2560 x 2048 | reg | 5,368,709,120 | 173,015,040 | 24 | 276 | 282.46 | 2.14 | 272.20 | 0.572 | chosen |
| 512 x 2560 x 2048 | reg-nok | 5,368,709,120 | 173,015,040 | 19 | 281 | 270.35 | 2.23 | 259.00 | 0.601 | REFUSED by the gate |

The problem itself, and what measurement alone would have picked:

| m x n x k | useful FLOPs 2 m n k | floor bytes 4 (m k + k n + m n) | roofline time (microseconds) | fastest by measurement alone | its lead over the choice (standard errors of the difference) | engine's choice | choice over the admitted runner-up (standard errors of the difference) |
|---|---:|---:|---:|---|---|---|---:|
| 1 x 2560 x 2048 | 10,485,760 | 20,989,952 | 32.49 | tiled | - | tiled | 73.6 |
| 48 x 2560 x 2048 | 503,316,480 | 21,856,256 | 33.83 | reg-nok | reg-nok by 2.5 | reg | 11.0 |
| 200 x 2560 x 2050 | 2,099,200,000 | 24,680,000 | 60.90 | reg-nok | reg-nok by 4.3 | reg | 92.1 |
| 512 x 2560 x 2048 | 5,368,709,120 | 30,408,704 | 155.75 | reg-nok | reg-nok by 3.9 | reg | 75.7 |
<!-- /worked -->

The engine's choices are `tiled` for the single row, where 20.99 million bytes must be read whatever the kernel and the larger tile only wastes lanes, and `reg` for the three taller shapes, with fractions of the measured roofline of 0.843, 0.328, 0.492 and 0.572. Read the second table: measurement alone would have chosen `reg-nok` at all three shapes where `reg` is chosen, by 4 to 5 percent of the mean, which is 2.5, 4.3 and 3.9 standard errors of the difference. A selector that only times caches that answer for the bucket and computes garbage whenever $k$ is not a multiple of 8; the gate is the only part of the pipeline that knows the difference. The same table is the honest statement of how far the engine is from the device. At $48 \times 2560 \times 2048$ the best candidate reaches 0.328 of the roofline, and the reason is in the work columns, not in the clock: the tile of `reg` is 128 rows, the problem has 48, so the kernel executes 1.34 billion FLOPs for a problem that needs 0.50 billion. Selection chooses among the candidates that exist; a fraction this low names a missing candidate, a tile of fewer rows, not a failure of the choice.

This is the stage at which the engine stops assuming and starts asking: which kernels may run on this device, which are right on this device, and which of those is cheapest by a measurement that was taken the way Chapter 11 says. Everything a larger engine adds (a cache of the answers, more candidates, a tuner that searches schedules) is a refinement of this pipeline.

Filtering by the name of the device ("the CPU never runs barrier kernels") instead of by the record stays true only until a new device appears. Timing the first run, or the mean of all 300, is a second mistake: the transient is 80 percent over the floor at run 0 and the compile adds twenty floors, and the mean of all runs differs from the steady mean by more than the gap that is being resolved. Comparing two means without the standard error of their difference chooses between candidates whose gap is noise. Measuring a candidate that has not passed the gate and recording its fraction as the engine's is another, since a fraction is a statement about a kernel that gives the right answer. Finally, reporting the time and not the work leaves a number nobody can check, where a fraction of the roofline from a stated roofline can be checked.

## Acceptance tests

The tests are the blocks printed in the milestones above, collected here as a checklist. An implementation passes when every statement holds. "Exact" means integers, digests and text equal; a tolerance is absolute unless stated.

| milestone | test | expected | tolerance |
|---|---|---|---|
| 1 | storage positions of the four views; flat order | as printed | exact |
| 1 | digests of A, B and the oracle product on the five integer cases; the smallest case in full | as printed | exact |
| 1 | product through the transposed view equals the product with the transpose copied out | identical digests | exact |
| 2 | naive kernel on the integer cases: workgroups, invocations, words loaded and stored per buffer, load and store blocks, $C$ equal to the oracle | as printed | exact |
| 3 | tiled and blocked kernels on the two ragged products: loads per buffer, loads in all equal the formula, stores, blocks | as printed | exact |
| 3 | naive, tiled and blocked kernels on the real-valued $13 \times 10 \times 37$ product | one digest for all three | exact |
| 4 | row sums and two-pass sums on both devices: loads, stores, digests or bits, $k_{\mathrm{rd}}$, outputs outside tolerance (zero) | as printed | exact for counts and bits; errors in $uS$ to 3 decimals |
| 4 | epilogue: words, bytes and blocks of the unfused and the fused design; saved words $= 4mn$; digests equal | as printed | exact |
| 5 | parity on three shapes, two kernels, two devices: $k_{\mathrm{rd}}$, tolerance, error, digest, entries equal on A and B | as printed | counts and digests exact; errors in $uS$ to 3 decimals |
| 5 | the unguarded kernel: passes where $4 \mid k$, fails with all $mn$ outputs outside tolerance where it does not; the guarded kernel passes all five | as printed | exact |
| 6 | quantized product: SQNR of the weights and of the output, relative error, $k_{\mathrm{rd}}$, outside tolerance (zero), digests, executor words, stored bytes | as printed | SQNR to 0.01 dB; counts and digests exact |
| 7 | capability filter on both units; gate verdicts; the four choices; fractions of the roofline | as printed | choices exact; fractions to 0.002; the means to 0.05 of the model's microseconds |

Three conventions apply to the whole list. The digests are reproduced only by a bit-exact generator and bit-exact float32 arithmetic (a product rounded and then a sum, or a fused multiply-add, as the device configuration says), so test in this order: the draws, the oracle digests (milestone 1), the integer kernels (milestones 2 and 3, which have no rounding), the real-valued kernels, and last the quantized and selected results. A digest that fails after all those before it passed points at an order of additions or at a library function, never at the generator. The model clock of milestone 7 uses `log` and `exp`, which the standard libraries do not round identically in the last bit; its means are therefore tested to the stated tolerance and its choices exactly, and the choices have margins of eleven standard errors or more over the admitted runner-up. And where the table shows the planted defect, the verdict is the test, not the number of the worst ratio.

A compact fingerprint of a correct engine, to check at the end:

<!-- worked: project_tensor_engine.acceptance_fingerprint -->
| milestone | quantity | value | tolerance |
|---|---|---|---|
| 1 | digest of the oracle product C, 2 x 2 x 3 (binary64) | 7544addf15e64632 | exact |
| 2 | naive kernel, 13 x 11 x 9: loads of A and B, stores of C | 2,574, 143 | exact |
| 3 | loads at 13 x 10 x 7: naive, tiled T = 4, blocked T = 8 | 1,820, 553, 322 | exact |
| 4 | two-pass sum of Y: float32 bits on device A and on device B | 0xc13b405b, 0xc13b405a | exact |
| 5 | parity 21 x 18 x 100, tiled: entries equal on A and B | 51 of 378 | exact |
| 5 | unguarded tiled kernel at 13 x 10 x 7: outputs outside tolerance | 130 of 130 | exact |
| 6 | int4 and int8, G = 32: output SQNR (dB) of the factored kernel | 20.22, 45.02 | 0.01 |
| 7 | choice for 1 x 2560 x 2048, 48 x 2560 x 2048, 200 x 2560 x 2050, 512 x 2560 x 2048 | tiled, reg, reg, reg | exact |
| 7 | fraction of the measured roofline of the choice at 512 x 2560 x 2048 (median of the steady runs) | 0.572 | 0.002 |
<!-- /worked -->

## The engine's counterpart in brain's backend layer

The engine of this chapter is the structure of brain's backend layer with the scale removed. `crates/backend-api/src/lib.rs` holds the `Backend` trait, the seven operations of milestone 2 with opaque handles; commit 7c8c97045 (July 2026) moved the three existing backends behind it with no change to the public API, so that a new backend is a crate that depends on the interface and nothing else. `crates/backend-cpu/src/lib.rs` is the CPU backend, and `crates/wgsl-cpu/src/lib.rs` is its kernel compiler, the executor of milestone 2 emitted as machine code. Commit 3f489e789 (September 2026) is the story of the two rules the executor has to keep: it found, by comparing that compiler with a second one over the same kernel source, that an early return was scoped to the first loop only and that workgroup memory was reused without clearing between workgroups; knowledge #112 draws the lesson, and `crates/wgsl-cpu/tests/it/workgroup_semantics.rs` is the test. The kernels of milestones 2 and 3 are `crates/kernels/wgsl/matmul.wgsl`, `matmul_tiled.wgsl` and `matmul_reg.wgsl` in the same directory, the naive, the $32 \times 32$ and the $128 \times 128$ kernels of milestone 7, with the shapes at which each is chosen in `crates/backend-api/src/select.rs`. The reference gate, the capability declarations and the measured fraction of the roofline are in `crates/gpu-core/src/native_tune.rs`; commit 72b363135 (October 2026) introduced the table of per-device choices for one quantized product, with each candidate declaring the capability it needs, a gate against a float64 oracle before any timing, and the achieved fraction of the device's own bound recorded beside the choice. What differs from this project is scale and execution, not the contract: brain's kernels run on real devices, its roofline comes from probes on the device, and its tolerance for a quantized candidate is derived from the formats of its inputs, the extension of milestone 5 to a kernel whose own inputs are rounded.

## Summary

If every acceptance test passes you have built and checked: a strided tensor and a float64 oracle; a backend with an executor that runs a kernel as the program of one invocation, counts the words and transactions it touches and refuses a barrier that some invocations skip; a naive, a tiled and a register-blocked matrix product that are correct on every ragged shape and whose global loads you can state in advance as exact integers, bit-identical to one another because tiling leaves the order of the additions alone; row sums and a two-pass sum with tolerances derived from the shape of the tree; a fused epilogue that removes exactly $4mn$ words of traffic and not one bit of the result; a second device with another tile, another order of additions and fused multiply-adds, held to a tolerance that is computed, together with a test that rejects a kernel that is correct only on tile-sized shapes; group-wise int8 and int4 weights decoded inside the product, with the error to expect and the error that must not occur; and a selection that filters by capability, gates by reference and chooses by a measurement that waited, warmed up and repeated enough, reporting work and the fraction of a roofline instead of a time.

What the project established, in order. A tensor is storage plus a recipe for reading it, and an oracle that is more accurate than anything it judges is the base of every test. The edges of a grid are where kernels fail, and a test is evidence only if its shapes contain them. Tiling changes the traffic and not the arithmetic; changing the arithmetic (lanes, fused multiply-adds, the shape of a tree) changes the bits and needs a tolerance, which has a derivation. A candidate must be right before it may be fast, and "right" is a gate run on the device, on shapes with ragged edges. A cost is work, and the honest statement of a speed is the work and the fraction of a measured roofline.

Where this stops. The engine runs on a simulator, so it says nothing about the speed of any real machine: its clock is a model, its devices are two configurations of one executor, and its candidates are four. It has no attention, no convolution, no scan, no sort, no packed codes (the simulator holds one code per word), no cache of selections and no search over schedules. Every one of these is an addition to the pipeline and not a change to it. The next volume leaves the machine and turns to text, the first kind of data whose representation is not already numbers; when attention arrives in Volume VII it needs the tiled product, the reductions and the fused epilogue of this chapter exactly as written here.

## Extensions

These are optional. Each is a bigger test of the same ideas.

1. **A real GPU backend.** Implement the seven operations over any compute API, translate the kernels of milestones 2 to 4 into its shading language, and run the parity test of milestone 5 with the device's own record in `capabilities()`. Report $k_{\mathrm{rd}}$ for the order your device's compiler chose (found by the test, not assumed) and any kernel for which bit-identity fails.
2. **A JIT.** Generate source text for a kernel's phase list instead of interpreting it, compile it once per (kernel, tile, device) and cache it under a key of the fields that change the program (Chapter 9). Show that editing the kernel changes the key.
3. **A cache-blocked CPU kernel on real hardware.** Write the blocked kernel as native code with a register block, measure it with the protocol of milestone 7 (stamps or a waited bracket, warm-up detection, at least 264 steady runs), and report it as a dated observation of a named machine: the FLOPs, the bytes, and the fraction of a roofline you measured with the probes of Chapter 10. Keep the time out of the claim.
4. **The missing second barrier.** Plant the other defect of Chapter 6, a compute phase and the next load phase merged. For which shapes does the parity test catch it, and does the order in which the executor runs the invocations of a phase change the count of wrong outputs?
5. **Split-$k$ as a candidate.** Add a kernel that deals the $k$ axis over two workgroups and combines the partials in a second dispatch. Derive its $k_{\mathrm{rd}}$, admit it through the gate with that value, and find the shapes at which the cost model of milestone 7 prefers it.
6. **A selection cache.** Key the choice of milestone 7 by the device record, a digest of the kernel sources, the row bucket and $n$, $k$ (Chapter 10). Edit one kernel and show that the old entry stops applying. Plant a defect that only the unbucketed shape exposes and say what the cache has to forget.
7. **Quantized candidates.** Add the factored kernel of milestone 6 as a candidate whose valid shapes are those with $G \mid k$, gate it against the float64 product of the decoded weights, and decide from the cost model at which $m$ the saved bytes matter.
8. **Tolerances with and without teeth.** Replace the derived tolerance in the parity test by a fixed $10^{-6}$ and by $10^{-3}\,S$. For the three shapes of milestone 5 and a kernel that drops the last term of every dot product, tabulate the false alarms of the correct device and the detections of the defective one (Chapter 9), and say which of the three tests a reviewer would accept.

## Answers

**Recall questions.**

1. The transpose is the same storage with the shape $[4, 3]$ and the strides $(1, 4)$ exchanged, and the offset stays 0. Element $[1, 2]$ is at storage position $0 + 1 \cdot 1 + 2 \cdot 4 = 9$, which holds the number 9.
2. The grid has $\lceil 10 / 4 \rceil = 3$ workgroups across and $\lceil 13 / 4 \rceil = 4$ down, 12 in all. A workgroup whose tile lies fully inside the output reaches the barrier with all 16 invocations: those are the $2 \times 3 = 6$ workgroups of columns 0 and 1 and rows 0 to 2. The other 6 (the last column tile, which holds columns 8 and 9 of 12, and the last row tile, which holds row 12 of 16) have invocations that returned before the barrier and others that did not, and the executor reports the error.
3. Tiled with $T = 4$: $k(m\lceil n/T \rceil + n\lceil m/T \rceil) = 7 (13 \cdot 3 + 10 \cdot 4) = 7 \cdot 79 = 553$. Naive: $2mnk = 2 \cdot 13 \cdot 10 \cdot 7 = 1{,}820$. Blocked with $T = 8$: $7(13 \cdot 2 + 10 \cdot 2) = 322$.
4. A left fold of 100 products: $k_{\mathrm{rd}} = 100$ and the bound is $100\,u S/(1 - 100\,u)$, which is $100.0$ in units of $uS$ to four digits. A tree of 128 products: $k_{\mathrm{rd}} = 1 + \log_2 128 = 8$, so $8.0$. A test may hold a program to the bound of the order it runs: the tree adds fewer roundings along its longest path, and a device whose order is fixed to a tree can be held to the tight bound. A test that used the left fold's bound would have no teeth for a tree.
5. The clock measured the cost of the host enqueueing the dispatch, which returns before the device has started. To make it the cost of the kernel, wait for the device (or read the device's own stamps) and discard the warm-up runs before taking the statistic.
6. 4 bits plus $32/32 = 1$ bit of scale is 5 bits per weight. $128 \cdot 24 = 3072$ weights take $3072 \cdot 5 / 8 = 1{,}920$ bytes, against $3072 \cdot 4 = 12{,}288$ in float32, a factor of 6.4.
