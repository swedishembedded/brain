<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Choosing the Fastest Correct Kernel

*How does an engine run near the roofline on hardware nobody tuned it for?*

A matrix product $C = A B$, with $A$ of shape $m \times k$ and $B$ of shape $k \times n$, can be run by three of the kernels of [Chapter 6](06-matmul.md): the naive one (one output per invocation, a serial loop over $k$), the tiled one (a $32 \times 32$ tile of outputs per workgroup, each invocation holding a $4 \times 4$ block) and the register-blocked one (a $128 \times 128$ tile, an $8 \times 8$ block per invocation). They compute the same function. The engine has to pick one for every product it runs, on a GPU that nobody has measured it on, and the shapes change from one request to the next: $m$ is 1 for one token and 512 for a prefill.

[Chapter 3](03-roofline.md) gives a way to rank them without running anything: count the bytes each kernel requests and the FLOPs of the product, and take the cost the roofline model assigns. The register-blocked kernel reuses every loaded number the most, so it requests the fewest bytes, and the model ranks it first at every shape. Here is that ranking next to a seeded model of a real device (the formulas are in the section after the next; the numbers are of the model GPU of Chapters 3 and 4, the one whose silicon reaches 36,000 (GFLOP/s) and 664 (GB/s)). Each entry is the cost of the problem's floor, every operand moved once and $2mnk$ FLOPs, divided by the cost of the kernel, so 1 would be the roofline itself:

<!-- worked: kernel_selection.model_vs_device -->
The model GPU unit of the next sections: peak 36,000 (GFLOP/s), bandwidth 664 (GB/s), so the ridge is 36,000 / 664 = 54.22 FLOP per byte. Three candidates, C[m, n] = A[m, k] B[k, n] in fp32. 'Predicted' is the roofline model of Chapter 3: the cost of the candidate is max(its bytes, 2 m n k / ridge), and the fraction is the cost of the problem's own floor (every operand once) divided by it. 'Device' is the model device of this chapter: the minimum of 8 seeded runs, as a fraction of the same floor. 'Lost' is 1 minus (device fraction of the model's pick over device fraction of the device's pick). The model device (its formulas are derived in the section 'What the model sees and what it does not'; the table needs only these parameters): 10,240 lanes, memory latency 200 ns, so n_need = 664 x 200 / 4 = 33,200 loads in flight keep the memory busy, one dispatch alone costs L = 5 microseconds x 664 (GB/s) = 3,320,000 bytes. Candidate (tile T_m x T_n, invocations per workgroup, register block r x s, loads in flight per invocation, issue efficiency): naive (1 x 1, 64, 1 x 1, 2, 1/5), tiled (32 x 32, 64, 4 x 4, 8, 16/(16 + 4 + 4 + 2) = 0.6154), reg (128 x 128, 256, 8 x 8, 8, 64/(64 + 8 + 8 + 2) = 0.7805). Bytes requested N_byte = 4 (m k t_n + k n t_m + m n) with t_m = ceil(m / T_m), t_n = ceil(n / T_n) (for naive T_m = T_n = 1); executed FLOPs 2 (t_m T_m)(t_n T_n) k for the tiled kernels and 2 m n k for naive; invocations t_m t_n n_wg for the tiled kernels and m n rounded up to a multiple of 64 for naive; o_cmp = min(1, invocations / 10,240), o_mem = min(1, invocations x loads in flight / 33,200); device cost B_dev = L + max(executed FLOPs / (ridge x e x o_cmp), N_byte / o_mem) with ridge 54.2169 = 36,000 / 664; floor B_fl = max(4 (m k + k n + m n), 2 m n k / ridge). A run costs B_dev x (1 + 0.04 u0) x (1 - 0.25 ln(1 - u2) if u1 < 0.2, else 1), three uniforms u0, u1, u2 per run in that order.

Generator: the book's SplitMix64, seed 0xbc12f335dfcc1d23 (`seed_of` of the name `kernel_selection.ranking`, defined on the pseudocode page). Draw order: for each shape in the order of the table, for each candidate in the order naive, tiled, reg, 8 runs of three uniforms each (u0, u1, u2).

| m x n x k | model's pick | its predicted fraction | device's pick | its device fraction | device fraction of the model's pick | lost |
|---|---:|---:|---:|---:|---:|---:|
| 1 x 2560 x 2048 | reg | 0.993 | tiled | 0.833 | 0.313 | 0.625 |
| 32 x 2560 x 2048 | reg | 0.812 | tiled | 0.471 | 0.320 | 0.321 |
| 129 x 2560 x 2048 | reg | 0.387 | reg | 0.367 | 0.367 | 0.000 |
| 512 x 2560 x 2048 | reg | 0.572 | reg | 0.557 | 0.557 | 0.000 |
| 512 x 100 x 4096 | reg | 0.676 | tiled | 0.157 | 0.077 | 0.508 |
<!-- /worked -->

At $m = 1$ the model picks the register-blocked kernel and predicts 0.993 of the roofline. On the device it reaches 0.313, and the tiled kernel, which the model ranked second, reaches 0.833: choosing as the model says gives up 62.5 percent of what was available. At $m = 512$ the model is right. At $512 \times 100 \times 4096$, a product with only 100 output columns, it is wrong again and the choice loses 50.8 percent. The roofline model of Chapter 3 is working as designed here. It is a bound: it says no kernel can cost less than its own bytes and FLOPs. It does not say how far above its bound each kernel lies, and that distance differs from kernel to kernel and from shape to shape, so a ranking of bounds is not a ranking of costs. Knowledge #15 met the same fact on one card: a selection rule that sent short matrices to the naive kernel was the bottleneck of a whole forward pass, and the right threshold was found by a sweep, not by arithmetic.

The sentence "run them and look" is the right answer, and it has five ways of going wrong, each of which becomes one stage of the engine's answer:

1. A candidate may not be able to run on the device at all. A kernel that needs workgroup barriers must not even be tried on a device whose compiler cannot execute them (a capability declaration, a correctness gate).
2. A candidate may run and compute the wrong numbers, and a wrong kernel is often the fastest, because it did less (a reference gate).
3. A measurement is noisy. Taking the first number, or the mean of a few, picks by luck (robust statistics, and a rule for ignoring differences inside the noise).
4. A measurement costs device work, so the answer must be kept and reused where it still holds and forgotten where it no longer does (a selection cache).
5. The result is only meaningful against a denominator that belongs to the device, not to its data sheet (a measured roofline).

The kernels are the candidates the stages work on. The figure shows the stages, with the counts of one request on the GPU unit and of the same request on the CPU unit:

![Two rows of five boxes: candidates, capability filter, reference gate, measure and selection cache. On the GPU unit four candidates enter, four pass the capability filter, the reference gate refuses one (reg-notail, wrong at m = 129) and takes 6 launches, the race takes 28 launches and drops tiled and naive, and reg is stored in the cache. On the CPU unit the capability filter leaves one candidate of four, the gate and the race have nothing to do, and naive is stored.](img/selection-pipeline.svg)

All the numbers of this chapter come from one seeded model, `scripts/book/worked/kernel_selection.py`, and nothing in it times a real machine. Costs are in the equivalent bytes of [Chapter 8](08-fusion.md): the roofline time multiplied by the bandwidth, so that a cost carries no seconds and holds for any machine with the same ridge point.

The stages ahead lean on six earlier ideas, from the roofline to the standard error. Bring them back from memory; the answers are at the end of the chapter.

1. (Volume IV, Chapter 3, the roofline.) A computation does $2 \times 10^9$ FLOP on 100 MB. On a device with $F_{\mathrm{pk}} = 40{,}000$ GFLOP/s and $\mathrm{BW} = 800$ GB/s, what is its intensity, on which side of the ridge point is it, and what is its cost $B_{\mathrm{eq}}$ in equivalent bytes (Chapter 8)?
2. (Volume IV, Chapter 6, the tile.) A $128 \times 128$ output tile is used for a product with $m = 1$ row. What share of the tile's outputs is real?
3. (Volume IV, Chapter 4, the 40,000 invocations.) How many invocations, each with one 4-byte load in flight, keep the memory of the model GPU busy, and why?
4. (Volume IV, Chapter 9, the capability record.) What does the record report for a capability the backend cannot establish, and why that value?
5. (Volume IV, Chapter 9, the tolerance.) A left fold over $K = 8$ products with $S = 4$: what is the tolerance $k_{\mathrm{rd}} u S/(1 - k_{\mathrm{rd}} u)$ in float32, to two significant digits?
6. (Volume I, Chapter 12, the standard error.) Five measurements have a sample standard deviation of 0.01. What is the standard error of their mean?

## Implementation candidates: an implementation and its conditions

A selector chooses among things, and every later stage is a statement about one of them, so we begin by saying what one is. An **implementation candidate** is one implementation of an operation together with the conditions under which it is valid: the capabilities a device must declare for it to run, the shapes it serves, and the contract of its result (the formats it reads and the tolerance its output is held to). The operation is the same for all candidates of a slot; the conditions are what differ.

The three candidates of the opening, with the counts that Chapter 6 taught. The issue efficiency is the fused multiply-adds of one step of $k$ over all the instructions of that step, so a kernel that spends its instructions on loads has a low one:

<!-- worked: kernel_selection.candidate_table -->
Three implementations of C = A B. 'Tile' is the block of outputs one workgroup owns (the naive kernel owns one output per invocation). 'Loads in flight' is the independent 4-byte loads an invocation has outstanding before it must wait: 2 for the naive kernel (the two operands of one step of k), and the staging loads of a step for the tiled kernels (a 32 x 8 slab of A and an 8 x 32 slab of B over 64 invocations is 8; a 128 x 8 and an 8 x 128 over 256 is 8). The issue efficiency e = r s / (r s + r + s + 2) counts the fused multiply-adds of one step of k against the reads of the slabs and 2 instructions of loop overhead; the naive kernel does 1 multiply-add against 2 loads from memory and the same overhead, 1 / 5.

| candidate | tile | invocations per workgroup | register block | loads in flight | issue efficiency e |
|---|---:|---:|---:|---:|---:|
| naive | 1 x 1 | 64 | 1 x 1 | 2 | 0.2000 |
| tiled | 32 x 32 | 64 | 4 x 4 | 8 | 0.6154 |
| reg | 128 x 128 | 256 | 8 x 8 | 8 | 0.7805 |
<!-- /worked -->

Two properties of the list matter. The first is a **reference**: one candidate is valid on every device and every shape, needs nothing, and defines what correct means. Here it is the naive kernel, as in brain, where the portable per-element kernel is "always present, always correct, the baseline every other family is checked against". Every other candidate is held to it, and a slot that loses all its other candidates still has one. The second is an **order**: the candidates are listed best guess first, here by the roofline model's predicted cost, because the model is a good guess even though it is not a ranking (it is right on half the shapes of the table below, and it costs nothing to ask). The head of the list is what the engine runs when nothing has been measured, and it is the incumbent that a measured challenger must beat.

The candidates as records of the book's pseudocode, with the conditions as data and not as code scattered through the selector:

```text
record Needs:
    barriers: bool            # workgroup barriers execute correctly
    workgroup_kib: int        # workgroup memory per workgroup
    workgroup_size: int       # invocations per workgroup
    matrix_unit: bool         # a matrix unit (the tensor core of Chapter 6) with the candidate's (a, b, accumulator) formats

record Candidate:
    name: text
    needs: Needs
    min_multiple: int         # m, n and k must be multiples of this (1: any shape)
    source: text              # the kernel text, whose digest keys the cache
```

A candidate for a different contract, a matrix product of fp16 operands on a matrix unit, is a candidate of another operation: its tolerance is that of fp16 inputs, Chapter 6's $2^{-11}$ per input, and it enters the list of a slot whose contract says fp16 operands. Mixing contracts in one list would let the fastest, least accurate kernel win a slot whose callers expect the fp32 answer.

## What the model sees and what it does not

The roofline model remains useful as an order, as a bound and as the unit in which results are stated, so it is worth writing down exactly what it counts and what the device does beyond that. Then its failure in the opening table has a cause.

For a candidate $c$ with an output tile of $T_m \times T_n$ and a shape $(m, n, k)$, the tile grid is $t_m = \lceil m / T_m \rceil$ by $t_n = \lceil n / T_n \rceil$. Each tile reads its slab of $A$ and of $B$ once, so the bytes requested are

$$N_{\mathrm{byte}}(c) = 4\,(m k\, t_n + k n\, t_m + m n),$$

the formula of Chapter 6 for a rectangular product (the naive kernel is the case $T_m = T_n = 1$, which requests $\approx 8 m n k$ bytes). The model's cost of the candidate is the cost of Chapter 8 with the candidate's own bytes and the useful FLOPs $N_{\mathrm{flop}} = 2mnk$:

$$B_{\mathrm{pred}}(c) = \max\bigl(N_{\mathrm{byte}}(c),\; N_{\mathrm{flop}} / \mathcal{I}_{\mathrm{ridge}}\bigr).$$

The cost of the problem itself, whatever the kernel, is the floor $B_{\mathrm{fl}} = \max(4(mk + kn + mn),\; N_{\mathrm{flop}}/\mathcal{I}_{\mathrm{ridge}})$ (every operand once), and the quantity in which every result is stated is

$$\mathrm{frac}(c) = \frac{B_{\mathrm{fl}}}{B(c)} \in (0, 1],$$

the achieved fraction of the roofline of Chapter 3, for the model's $B_{\mathrm{pred}}$ and for the device's $B_{\mathrm{dev}}$ below.

What the device does beyond the model is four things, each of which Chapters 4, 6 and 8 taught and none of which the model's two counts contain.

*Idle lanes in a tile.* The hardware runs whole tiles. The FLOPs it executes are $N_{\mathrm{flop}}^{\mathrm{pad}} = 2\,(t_m T_m)(t_n T_n)\,k$, which for one row in a $128 \times 128$ tile is 128 times the useful FLOPs (the real share of Chapter 6's table, 0.8 percent).

*Invocations that exist.* A workgroup of $n_{\mathrm{wg}}$ invocations per tile gives $n_{\mathrm{inv}} = t_m t_n n_{\mathrm{wg}}$ invocations (the naive kernel rounds $mn$ up to a multiple of 64). The arithmetic is spread over at most the lanes of the device, so only $o_{\mathrm{cmp}} = \min(1, n_{\mathrm{inv}} / N_{\mathrm{lane}})$ of the arithmetic units work, with $N_{\mathrm{lane}} = 10{,}240$ lanes in all. The memory needs $n_{\mathrm{need}} = \mathrm{BW}\, t_{\mathrm{lat}} / 4$ loads of 4 bytes in flight (Chapter 4's count: 33,200 at this unit's 664 (GB/s) and 200 ns), and an invocation keeps $n_{\mathrm{fl}}$ of them, so $o_{\mathrm{mem}} = \min(1, n_{\mathrm{inv}}\, n_{\mathrm{fl}} / n_{\mathrm{need}})$.

*The instruction mix.* Only the share $\mathrm{eff}_c$ of the issue slots (the issue efficiency of the table) is a fused multiply-add.

*The start.* One dispatch alone costs $L = t_0 \mathrm{BW}$ (Chapter 8): 5 microseconds at 664 (GB/s) is 3,320,000 bytes.

Together:

$$B_{\mathrm{dev}}(c) = L + \max\left(\frac{N_{\mathrm{flop}}^{\mathrm{pad}}}{\mathcal{I}_{\mathrm{ridge}}\;\mathrm{eff}_c\; o_{\mathrm{cmp}}},\;\; \frac{N_{\mathrm{byte}}(c)}{o_{\mathrm{mem}}}\right).$$

Every factor in this formula is at most 1 in the denominator or at least 1 in the numerator, so $B_{\mathrm{dev}} \ge B_{\mathrm{pred}} \ge B_{\mathrm{fl}}$: the model is a lower bound on each candidate, as Chapter 3 said it was. What differs between candidates is the gap. One shape, $77 \times 2560 \times 2048$ (the 77 text tokens of knowledge #15), shows the terms:

<!-- worked: kernel_selection.cost_model_example -->
The shape m = 77, n = 2560, k = 2048 on the model GPU unit. Ridge 54.2169; the start of one dispatch alone costs 3,320,000 bytes (5 microseconds times a bandwidth of 664 (GB/s)); the invocations that keep the memory busy at one load in flight each: 33,200; lanes 10,240. The floor of the problem is 22,390,784 equivalent bytes.

| candidate | tile grid | invocations | bytes requested | FLOPs executed | o_cmp | o_mem | predicted cost | device cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| naive | 77 x 2560 | 197,120 | 3,230,402,560 | 807,403,520 | 1.0000 | 1.0000 | 3,230,402,560 | 3,233,722,560 |
| tiled | 3 x 80 | 15,360 | 114,165,760 | 1,006,632,960 | 1.0000 | 1.0000 | 114,165,760 | 117,485,760 |
| reg | 1 x 20 | 5,120 | 34,375,680 | 1,342,177,280 | 0.5000 | 1.0000 | 34,375,680 | 66,756,518 |
<!-- /worked -->

The register-blocked kernel has a tile grid of $1 \times 20$, which is 5,120 invocations on 10,240 lanes: $o_{\mathrm{cmp}} = 0.5$, and its device cost, 66,756,518, is nearly twice its predicted 34,375,680. The tiled kernel has 15,360 invocations, fills the lanes and the memory, and its device cost is within 3 percent of its prediction. On this shape the model's ranking and the device's agree, but the margin between the two kernels is not what the model says. Where $m$ is smaller the register-blocked kernel's tile is almost empty and its grid smaller still, and the order flips. All sixteen shapes, with the model's fraction and the device's, each device fraction the minimum of 8 seeded runs of the noise of the next sections:

<!-- worked: kernel_selection.ranking_table -->
Each cell is predicted fraction / device fraction of the problem's roofline floor, the device fraction being the minimum of 8 seeded runs. Same model GPU unit, same runs as `kernel_selection.model_vs_device`.

Generator: the book's SplitMix64, seed 0xbc12f335dfcc1d23 (`seed_of` of the name `kernel_selection.ranking`, defined on the pseudocode page). Draw order: for each shape in the order of the table, for each candidate in the order naive, tiled, reg, 8 runs of three uniforms each (u0, u1, u2).

| m x n x k | naive | tiled | reg | model's pick | device's pick | lost |
|---|---:|---:|---:|---|---|---:|
| 1 x 2560 x 2048 | 0.500 / 0.076 | 0.970 / 0.833 | 0.993 / 0.313 | reg | tiled | 0.625 |
| 4 x 2560 x 2048 | 0.125 / 0.076 | 0.890 / 0.763 | 0.971 / 0.315 | reg | tiled | 0.587 |
| 8 x 2560 x 2048 | 0.063 / 0.062 | 0.803 / 0.709 | 0.944 / 0.316 | reg | tiled | 0.555 |
| 16 x 2560 x 2048 | 0.032 / 0.031 | 0.673 / 0.603 | 0.895 / 0.314 | reg | tiled | 0.480 |
| 32 x 2560 x 2048 | 0.016 / 0.016 | 0.510 / 0.471 | 0.812 / 0.320 | reg | tiled | 0.321 |
| 48 x 2560 x 2048 | 0.011 / 0.011 | 0.296 / 0.282 | 0.745 / 0.325 | reg | reg | 0.000 |
| 64 x 2560 x 2048 | 0.008 / 0.008 | 0.262 / 0.251 | 0.690 / 0.330 | reg | reg | 0.000 |
| 77 x 2560 x 2048 | 0.007 / 0.007 | 0.196 / 0.189 | 0.651 / 0.335 | reg | reg | 0.000 |
| 128 x 2560 x 2048 | 0.005 / 0.005 | 0.146 / 0.143 | 0.572 / 0.370 | reg | reg | 0.000 |
| 129 x 2560 x 2048 | 0.005 / 0.005 | 0.131 / 0.129 | 0.387 / 0.367 | reg | reg | 0.000 |
| 256 x 2560 x 2048 | 0.005 / 0.005 | 0.146 / 0.145 | 0.572 / 0.549 | reg | reg | 0.000 |
| 512 x 2560 x 2048 | 0.005 / 0.005 | 0.146 / 0.145 | 0.572 / 0.557 | reg | reg | 0.000 |
| 1024 x 130 x 2048 | 0.005 / 0.005 | 0.131 / 0.125 | 0.389 / 0.149 | reg | reg | 0.000 |
| 512 x 100 x 4096 | 0.006 / 0.006 | 0.171 / 0.157 | 0.676 / 0.077 | reg | tiled | 0.508 |
| 4096 x 1 x 4096 | 0.500 / 0.123 | 0.970 / 0.917 | 0.992 / 0.514 | reg | tiled | 0.439 |
| 1 x 100 x 100 | 0.507 / 0.003 | 0.971 / 0.009 | 1.000 / 0.006 | reg | tiled | 0.324 |

The model's pick differs from the device's on 8 of the 16 shapes.
<!-- /worked -->

![Fraction of the roofline floor against log10 of m for n = 2560 and k = 2048. The register-blocked kernel's model curve (dashed) starts at 0.99 and falls to 0.57 at m = 128 with a drop at 129, but its device curve (solid) is flat near 0.31 up to m = 77 and then climbs to 0.55. The tiled kernel's device curve starts at 0.83 and falls to 0.14. The naive kernel's device curve stays below 0.08.](img/candidate-fractions.svg)

The model's pick differs from the device's on half of them, always in the same direction: it overrates the big-tile kernel where the tile is mostly empty, which is where $m$ is small or $n$ is narrow. The crossing is a property of the shape, not a constant. The ratio of the two costs, with the model's prediction beside it, and the crossing as it moves with the output width:

![Log10 of the cost of the register-blocked kernel divided by the cost of the tiled kernel against log10 of m. The model's dashed curve stays below zero everywhere. The device's solid curve starts at 0.42 above zero for m = 1 and crosses zero near m = 33, marked by a vertical line, after which the register-blocked kernel is cheaper.](img/crossover-vs-m.svg)

<!-- worked: kernel_selection.crossover_table -->
The smallest m at which the register-blocked kernel's noise-free device cost is below the tiled kernel's (no randomness), by output width n and depth k, on the model GPU unit.

| n x k | crossover m |
|---|---:|
| 2560 x 2048 | 33 |
| 1024 x 2048 | 113 |
| 256 x 2048 | 481 |
| 130 x 2048 | 837 |
<!-- /worked -->

No threshold on $m$ is right for all $n$: the crossing moves from 33 at $n = 2560$ to 837 at $n = 130$, because a narrow output leaves most of a 128-wide tile idle whatever $m$ is. A rule of the form "tile from $m \ge 8$ and $n \ge 128$" (the shape of brain's rule, section below) is a fit to one device and one $k$; a different device with other lanes, other latency and another start cost has other crossings, which is the reason to measure instead of to tune a constant.

None of this makes the model useless. It gives the order of the list (right on 8 of the 16 shapes at no cost), it is a lower bound that no honest measurement may violate, and its floor is the denominator of $\mathrm{frac}$. What it cannot do is decide. The next four sections are the stages that decide.

## Capability declaration: what may run

Before anything is timed, the candidates that cannot run on the device must be removed, because a candidate that is wrong on a device is not a slow candidate and speed cannot tell the two apart.

A **capability declaration** is the statement of what a device can execute (the record of [Chapter 9](09-backends.md), queried from the device and absent when unknown) together with the statement of what each candidate needs of it. A candidate is **admissible** on a device when every need is declared. The CPU unit and the GPU unit of Chapter 3, with different ridge points (10 and 50 in the specification), differ here as they do in speed: the CPU unit's compiler runs a workgroup as loops and cannot execute a barrier kernel correctly, and it has no matrix unit.

<!-- worked: kernel_selection.capability_filter -->
The device side is the record of Chapter 9 (queried, an unknown capability is absent): the CPU unit declares no workgroup barriers (its compiler runs a workgroup as loops and cannot execute a barrier kernel correctly, Chapter 9) and no matrix unit; the GPU unit declares barriers, 16 KiB of workgroup memory, workgroups of up to 256 and a (f16, f16, f32) matrix unit (the tensor core of Chapter 6, which multiplies 16 x 16 x 16 fragments with f32 accumulation; its kernel also uses subgroups, the groups of lanes that execute together, Chapter 4).

| candidate | needs | valid for | CPU unit | GPU unit |
|---|---|---|---|---|
| naive | nothing | any shape | runs | runs |
| tiled | workgroup barriers, 2 KiB workgroup memory | any shape (row and column guards) | refused: no workgroup barriers | runs |
| reg | workgroup barriers, 8 KiB workgroup memory, workgroups of 256 | any shape (row and column guards) | refused: no workgroup barriers | runs |
| coop16 | matrix unit with (f16, f16, f32), subgroups | m, n, k all multiples of 16; inputs already fp16 | refused: no matrix unit | runs |

The filter is a correctness gate, not a preference. What the roofline model alone would pick on the CPU unit (ridge 10.67), for m = 512, n = 2560, k = 2048, if the filter did not exist:

| candidate | FLOP per byte requested | predicted fraction of the floor | executable on the CPU unit |
|---|---:|---:|---:|
| naive | 0.25 | 0.023 | yes |
| tiled | 7.94 | 0.744 | no |
| reg | 31.03 | 1.000 | no |

The shape condition of coop16 on four shapes:

| m x n x k | coop16 |
|---|---|
| 64 x 64 x 64 | valid |
| 77 x 2560 x 2048 | invalid (a dimension is not a multiple of 16) |
| 512 x 2560 x 2048 | valid |
| 48 x 2560 x 2048 | valid |
<!-- /worked -->

The filter is a correctness gate and the table's second half shows why it must not be a preference. If the roofline model alone chose on the CPU unit, it would pick the register-blocked kernel, with a predicted fraction of 1.000, a kernel that the unit cannot run. A selector that only ranks by cost would try it. A selector that treats the filter as a speed hint ("the CPU is slow at barriers, so rank it lower") would still try it when nothing else looked better. The rule is that no cost, measured or modelled, is ever computed for an inadmissible candidate. Three details follow from it.

*An unknown capability is absent.* The record reports what it cannot establish as the value that forbids the feature (Chapter 9), so a device that cannot be queried is served only by candidates that need nothing. The engine becomes cautious, never optimistic.

*A capability says that a feature executes correctly, not that it is fast.* A device may accept a half-precision kernel and run it at a small fraction of its single-precision rate; the declaration admits the candidate and the measurement, stage four, decides that it loses. Collapsing the two (admitting a feature only if it is known to be fast) would make a measurement the input of a correctness gate, and a gate whose answer depends on a measurement that has not yet run cannot be the first stage.

*A declaration must be checked against the kernel it describes.* A need written by hand is a claim, and a candidate whose declared floor is lower than what its text uses is offered to devices that cannot run it. brain checks its CUDA registry as data: a source that uses a four-way integer dot instruction cannot declare a floor below the generation that introduced it. The same check is cheap for any registry: scan the code, not the comments, for the instructions that imply a need, and compare with the declaration.

Shape conditions are part of the declaration. The matrix-unit candidate of the table needs $m$, $n$ and $k$ all multiples of 16, because a fragment load (the matrix unit of [Chapter 6](06-matmul.md) loads and multiplies a $16 \times 16$ block of each operand at once, a *fragment*) reads a whole block; on a device that has the unit, a shape of 77 rows is not served by it, and the selector's answer for that shape is computed from the other candidates. The reference needs nothing, so the filter never returns an empty list:

```text
function admissible(c: Candidate, caps: Caps, m: int, n: int, k: int) -> bool:
    need = c.needs
    if need.barriers and not caps.barriers:
        return false
    if need.workgroup_kib > caps.workgroup_kib or need.workgroup_size > caps.workgroup_size:
        return false
    if need.matrix_unit and not caps.matrix_unit:
        return false
    return m mod c.min_multiple == 0 and n mod c.min_multiple == 0 and k mod c.min_multiple == 0
```

## The measured roofline: the device's own denominator

Every result of the later stages is a fraction of the roofline, and a fraction of the wrong roofline is a number about nobody. So the device's two rates are measured, with two kernels whose work we can count.

The specification of the model GPU says 40,000 (GFLOP/s) and 800 (GB/s). A particular unit of it reaches less, as every real device does, and a kernel graded against the specification looks worse than it is by the same factor every time, so that rows of a profile that should be ranked against each other are all shifted and a kernel at its device's limit still looks unfinished. The two rates cannot be read from the device: no compute interface reports what the silicon achieves. They are measured, each with a kernel that does nothing else.

The **compute probe** is a chain of fused multiply-adds held in registers with no memory access in its loop: $2^{20}$ invocations, each with 8 independent accumulators updated $a \leftarrow \mathrm{fma}(a, c, d)$ for $\mathrm{trip}$ steps, 16 FLOPs per step. Eight independent accumulators keep the multiply-add pipeline from waiting on its own result however long that pipeline is; $c$ and $d$ arrive through the parameter block, so no compiler can fold the chain into a constant (a folded loop measures nothing and reports a spectacular rate); and the explicit fused instruction matters, since $a c + d$ written as two operations compiles on a backend that forbids contraction (Chapter 9) to two instructions per two FLOPs and halves the rate the probe reports. The **bandwidth probe** streams: $c \leftarrow c + s\,a$ over two buffers of $2^{26}$ elements, 12 bytes per element (two reads and a write), 805,306,368 bytes per pass, far beyond any cache.

The probe measures the silicon, not the best kernel we have: a compute roof derived from the register-blocked matrix product would grade the engine's kernels against the engine's best kernel and hide exactly the gap that the stages are there to expose.

Both probes need a run long enough that the start of the dispatch is a negligible part of it, and no fixed trip count does that on every device, so the probe grows its own: run the probe three times, take the minimum, accept the attempt when its cost is at least 100 launches (the start is then at most 1 percent), and otherwise scale the trip count to the target and at least double it. The rate is the work of the accepted attempt over its time:

```text
function calibrate_rate(run: function(int) -> real, work_per_trip: real, t_launch: real, trip0: int) -> (real, real):
    # run(trip) executes the probe for `trip` steps and returns its seconds.
    # Returns the measured rate (work per second) and the work all attempts executed.
    trip = trip0
    total = 0.0
    loop:
        t = min(run(trip), run(trip), run(trip))
        total = total + 3 * work_per_trip * trip
        if t >= 100 * t_launch:
            return (work_per_trip * trip / t, total)
        trip = max(2 * trip, ceil(trip * 100 * t_launch / t))
```

The calibration is a measurement like any other: it costs device work, and the work is countable. The two probes on the two units, with the model's noise (a run is its cost times a factor of at least 1, described in the section on measuring; three uniforms per run):

<!-- worked: kernel_selection.calibration -->
Two probes, as described above. Compute probe: 2^20 invocations, each a chain of 8 independent fused multiply-adds repeated `trip` times (16 FLOPs per step), starting at trip = 256. Bandwidth probe: c = c + s a over 2^26 elements (12 bytes per element, 805,306,368 bytes per pass), repeated `trip` passes, starting at 1. Each attempt runs 3 times and keeps the minimum; an attempt is accepted when its cost is at least 100 launches, otherwise trip becomes max(2 trip, ceil(trip x 100 launches / cost)). A run costs (launch + work / true rate) x its noise factor (three uniforms, as in the ranking). The hidden true rates are those of the units in `DEVICES` (GPU 36,000 and 664; CPU 640 and 60, in GFLOP/s and GB/s), launches are 5 and 2 microseconds. The rates are reported as the work of the accepted attempt divided by its time. Generators: the book's SplitMix64, seeds 0xa8e288b844cca4f1 (`kernel_selection.calibration.CPU`), 0x9d012f580657f66b (`kernel_selection.calibration.GPU`). Draw order: the compute probe's runs, then the bandwidth probe's, 3 uniforms per run, attempts in order.

| device | spec F (GFLOP/s) | measured F | % of spec | spec BW (GB/s) | measured BW | % of spec | spec ridge | measured ridge |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CPU | 800 | 627 | 78.4 | 80 | 60 | 74.6 | 10.00 | 10.50 |
| GPU | 40,000 | 34,469 | 86.2 | 800 | 646 | 80.8 | 50.00 | 53.32 |

The work the probes executed, all attempts together:

| device | compute attempts | GFLOP executed | bandwidth attempts | GB moved |
|---|---:|---:|---:|---:|
| CPU | 1 | 12.9 | 1 | 2.4 |
| GPU | 3 | 166.9 | 1 | 2.4 |
<!-- /worked -->

The GPU unit's measured rates, 34,469 (GFLOP/s) and 646 (GB/s), are 86.2 and 80.8 percent of the specification and give a ridge of 53.3, not 50: a different device has a different roofline even when it carries the same name, and the ridge point decides which kernels are bandwidth-bound. Both are a little below the unit's true rates (36,000 and 664), by 4.3 and 2.6 percent. That is how every measurement of a rate of work behaves: a run never beats the floor of the device, so the best of a few runs is above it, and the measured roofline is a slightly low roofline. A fraction slightly above 1 is therefore within the noise of the roof; one far above 1 is a defect of the measurement. The whole calibration of the GPU unit executed 166.9 (GFLOP) and moved 2.4 (GB), once, and is saved with the device's identity.

The **measured roofline** is the pair of rates $(F_{\mathrm{ms}}, \mathrm{BW}_{\mathrm{ms}})$ found by two such probes on the device itself. Results are then stated against it. The same best-of-8 runs of the opening table, graded four ways:

<!-- worked: kernel_selection.fraction_of_roofs -->
The same best-of-8 runs as `kernel_selection.ranking_table` (no new draws), graded against four rooflines: the unit's true rates (36,000 and 664: the fractions of the ranking table), the specification (40,000 and 800), the calibration of `kernel_selection.calibration` on this unit (34,469 and 646), and a calibration whose compute probe ran on an idle device and measured 14,244 (its bandwidth is the calibrated one). The bound of a row is max(2 m n k / F, 4 (m k + k n + m n) / BW). The cold probe is `kernel_selection.calibration.GPU.cold` (seed 0x1fca8e5bcdf55823): the compute probe of the calibration, with the device's true compute rate multiplied by 0.4.

| kernel, shape | of the true roofline | of the specification | of the measured roofline | of the cold-probe roofline |
|---|---:|---:|---:|---:|
| tiled, 1 x 2560 x 2048 | 0.833 | 0.692 | 0.856 | 0.856 |
| reg, 512 x 2560 x 2048 | 0.557 | 0.501 | 0.582 | 1.408 |
| tiled, 4096 x 1 x 4096 | 0.917 | 0.761 | 0.942 | 0.942 |
<!-- /worked -->

Against the specification the tiled kernel at $m = 1$ looks like 0.692 of the roof; against the measured roofline it is 0.856, within 3 percent of the unit's true 0.833. The specification builds a pessimism of 10 to 17 percent into these rows, and a team that ranks kernels by distance from the roof will chase the wrong ones. The last column is the opposite failure: a compute probe that ran on an idle device, which has not yet raised its clock, measured 14,244 (GFLOP/s), and the register-blocked kernel then appears to run at 1.408 of the roof, more than the silicon can do. A fraction above 1 beyond the noise of the probes is the signature of a bad roof. The probe must therefore run after a warm-up of continuous work, and a test that grades real work against the roof (it can never be above) catches a roof that was measured cold. A measured roof that is persisted poisons every later ratio of the machine, so its validity is worth a test.

A fraction measured against the device's own roofline is dimensionless and has no unit of time in it, so it is comparable between kernels, between shapes and between devices. It is also the honest output of selection. A choice is made among the candidates that exist; if the best of them reaches 0.31 of the roofline, selection cannot mend that, and a table that lists its classes under a stated fraction names the work that remains, a new candidate.

## Reference-gated selection: fast and right

A candidate that is faster because it computes less must never win, which calls for a check that runs on the device before anything is timed.

Here is a candidate that nobody wrote on purpose. The register-blocked kernel needs a guard so that a partial tile at the end of $m$ does not read or write outside the matrix. Remove the guard (an easy edit, and it makes the inner loop shorter) and the kernel becomes correct only when $m$ is a whole number of 128-row tiles: it computes the rows of the whole tiles and leaves the others at their initial value. Call it `reg-notail`. Three shapes of the bucket 129 to 256, with the device fractions of the three candidates, each the minimum of 8 seeded runs:

<!-- worked: kernel_selection.fast_and_wrong -->
Device fractions of the floor, minimum of 8 seeded runs, for three shapes of the bucket 129 .. 256. The cost of reg-notail is the cost of reg on its 128 x (m div 128) rows. Generator: the book's SplitMix64, seed 0x323749babe262744 (`seed_of` of the name `kernel_selection.fast_and_wrong`, defined on the pseudocode page). Draw order: per shape, per candidate in the order tiled, reg, reg-notail, 8 runs of 3 uniforms each.

| m x n x k | tiled | reg | reg-notail | fastest by measurement alone |
|---|---:|---:|---:|---|
| 129 x 2560 x 2048 | 0.128 | 0.368 | 0.374 | reg-notail |
| 200 x 2560 x 2048 | 0.137 | 0.475 | 0.563 | reg-notail |
| 255 x 2560 x 2048 | 0.143 | 0.549 | 0.735 | reg-notail |
<!-- /worked -->

By measurement alone the defective kernel wins all three, and the margin grows with the size of the tail it skips, from nothing at 129 (one row dropped, hidden by the noise) to 0.735 against 0.549 at 255, where it computes 128 rows and silently drops 127. A selector that only times will cache this as the answer for the bucket.

**Reference-gated selection** is selection in which a candidate may win only after its output on the device it will run on has matched a reference within a tolerance derived from its arithmetic. The gate has five rules, each of which has a way of going wrong without it.

*The tolerance is derived, per candidate, from the arithmetic* ([Chapter 9](09-backends.md)). A left fold of $K$ products is held to $k_{\mathrm{rd}} u S/(1 - k_{\mathrm{rd}} u)$ with $k_{\mathrm{rd}} = K$, a tree to the same expression with $k_{\mathrm{rd}} = 1 + \lceil \log_2 K \rceil$, each against the exact value of the sum. Bit-identity is demanded only of a candidate whose order is fixed by construction. A candidate that reorders the additions (a tree, a split over $k$) must be admitted, and a fixed tolerance chosen by looking at what passes would either reject it or admit a defect.

*Every checked output must pass,* not their root mean square. A bad tile edge or one wrong row passes an average over many good outputs and fails the maximum. brain's admission rule requires both.

*The positions include the edges.* The sampled positions are the first, middle and last rows and columns at tile edges ($0, 1, 15, 16, 17$, a third and half of $n$, and the last 17 and 1): a tail defect is found only by an output in the tail.

*The output is poisoned first.* The buffer is filled with a value that cannot be a correct answer (not-a-number) before the candidate runs, so that an output the candidate never stores is an error and not the value an earlier candidate left in the shared buffer. (The worked example below leaves the zero that a fresh buffer holds, the harder case, since 0 can be a correct output.)

*It runs on the device.* The arithmetic belongs to the device's compiler: contraction and the order of additions are decided there (Chapter 9), so a gate run on another machine does not test this one.

The four gate shapes, three candidates: the left fold `reg`, the tree `reg-tree` that adds the products in a tree of adjacent pairs, and `reg-notail`:

<!-- worked: kernel_selection.gate_table -->
The gate of the text: for each shape, the 27 output positions of rows {0, m div 2, m - 1} and columns {0, 1, 15, 16, 17, n div 3, n div 2, n - 17, n - 1} (duplicates removed), entries 2 uniform() - 1 rounded to float32, the products and sums of a candidate each rounded to float32 (no fused multiply-add). reg is a left fold over k; reg-tree adds the products in a tree of adjacent pairs; reg-notail is reg with its row guard removed, so it computes only the 128 x (m div 128) rows of whole tiles and leaves the other rows at their initial zero. The reference of a position is math.fsum of the exact products (a correctly rounded double sum); the tolerance is k_rd u S / (1 - k_rd u) with u = 2^-24, S = sum of |a b|, k_rd = k = 2048 for a left fold and 1 + log2(2048) = 12 for the tree. 'Worst' is the largest error divided by the tolerance. Generator: the book's SplitMix64, seed 0x2a5c91e9c44dc36e (`seed_of` of the name `kernel_selection.gate`, defined on the pseudocode page). Draw order: per shape in the order of the table, the entries of the sampled rows of A in increasing row order (k values each), then the entries of the sampled columns of B in increasing column order (k values each).

| m x n x k | candidate | positions checked | outside tolerance | worst error / tolerance | verdict |
|---|---|---:|---:|---:|---|
| 128 x 2560 x 2048 | reg | 27 | 0 | 0.000668 | admitted |
| 128 x 2560 x 2048 | reg-tree | 27 | 0 | 0.00731 | admitted |
| 128 x 2560 x 2048 | reg-notail | 27 | 0 | 0.000668 | admitted |
| 129 x 2560 x 2048 | reg | 27 | 0 | 0.000848 | admitted |
| 129 x 2560 x 2048 | reg-tree | 27 | 0 | 0.00841 | admitted |
| 129 x 2560 x 2048 | reg-notail | 27 | 9 | 699 | REFUSED |
| 200 x 2560 x 2048 | reg | 27 | 0 | 0.000292 | admitted |
| 200 x 2560 x 2048 | reg-tree | 27 | 0 | 0.00908 | admitted |
| 200 x 2560 x 2048 | reg-notail | 27 | 9 | 397 | REFUSED |
| 256 x 2560 x 2048 | reg | 27 | 0 | 0.000459 | admitted |
| 256 x 2560 x 2048 | reg-tree | 27 | 0 | 0.00816 | admitted |
| 256 x 2560 x 2048 | reg-notail | 27 | 0 | 0.000459 | admitted |
<!-- /worked -->

Both correct kernels pass everywhere, with errors far below their tolerance (the tree is 0.0091 at worst of its own, larger, allowance), and the defective one passes at 128 and at 256 and is refused at 129 and at 200, by factors of 699 and 397 over the tolerance. The shapes 128 and 256 are the largest members of two power-of-two buckets, which is where a tuner naturally measures. That is the limit of the stage and the reason the pipeline gates a bucket at its two extremes, the largest member and the smallest ($129$ for the bucket $129 \ldots 256$). A gate protects only the shapes it covers, so the cache entry must never be wider than the gate: every shape that maps to a key must be a shape whose class of failure (a full tile, a partial one) the gate has seen. And the tolerance, loose by two orders of magnitude here as in Chapter 9, finds structural defects, a missing tail, a wrong stride, a dropped row, not a last-bit difference; the model-level check of whole-network quality remains a separate gate.

```text
function positions(m: int, n: int) -> list<(int, int)>:
    rows = [0, m // 2, m - 1]
    cols = [0, 1, 15, 16, 17, n // 3, n // 2, n - 17, n - 1]
    out = []
    for r in rows:
        for c in cols:
            if r < m and c >= 0 and c < n and not any(p == (r, c) for p in out):
                append (r, c) to out
    return out

function gate(value: function(int, int) -> f32, a: array<f32>[M, K], b: array<f32>[K, N], k_rd: int) -> bool:
    # value(i, j) is what the candidate stored at C[i, j], read back from the device
    u = 2 ^ (-24)
    for (i, j) in positions(M, N):
        exact = exact_dot(a[i, :], b[:, j])
        S = sum(abs(a[i, p] * b[p, j]) for p in 0 .. K)
        tol = k_rd * u * S / (1 - k_rd * u)
        if not is_finite(value(i, j)) or abs(value(i, j) - exact) > tol:
            return false
    return true
```

In `gate`, `exact_dot(x, y)` is the exact value of the dot product of two vectors of floating-point numbers (computed in rational arithmetic, or with a correctly rounded accumulation, and never in the format under test; `docs/book/pseudocode.md` defines it).

The gate costs one launch of the candidate at each gate shape and a host-side reference of 27 outputs at most, $2K$ FLOPs each; the launches are counted with the measurements below.

## Measuring honestly enough: repeats, a robust statistic, noise

The stage that decides must not decide by luck. To see how luck enters, we model what makes a measurement noisy, compare four ways of reading a few runs, and add a rule for ignoring differences that are inside the noise.

The model device returns, for each launch, its noise-free cost times a factor drawn from three uniforms $u_0, u_1, u_2$:

$$\text{factor} = (1 + 0.04\,u_0)\,\bigl(1 - 0.25 \ln(1 - u_2)\ \text{if}\ u_1 < 0.2,\ \text{else}\ 1\bigr).$$

Every launch is lengthened by a jitter in $[0, 4\%)$, and one launch in five is hit by contention that lengthens it by 25 percent on average (an exponential). This is a model, and its one structural claim is the one that real measurements of a busy device share: noise only lengthens a run. Another process on the card stretches whatever it interrupts and shortens nothing, so the floor of the device is the best a run can do. Eight launches of an incumbent A (floor 1.00) and a challenger B whose floor is 3 percent lower:

<!-- worked: kernel_selection.noise_runs -->
One pair of candidates in the units of the incumbent's floor: A has floor 1.00, B has floor 0.97. Each run is floor x (1 + 0.04 u0) x (1 - 0.25 ln(1 - u2) if u1 < 0.2, else 1). Generator: the book's SplitMix64, seed 0x87eebb6e4d5931c9 (`seed_of` of the name `kernel_selection.noise_runs`, defined on the pseudocode page). Draw order: the 8 runs of A, then the 8 runs of B, 3 uniforms each.

| run | A (floor 1.00) | B (floor 0.97) |
|---|---:|---:|
| 0 | 1.0288 | 0.9742 |
| 1 | 1.0358 | 1.2529 |
| 2 | 1.0106 | 1.0085 |
| 3 | 1.0238 | 0.9754 |
| 4 | 1.0126 | 1.0584 |
| 5 | 1.0328 | 0.9979 |
| 6 | 1.0183 | 0.9961 |
| 7 | 1.0153 | 0.9853 |

Minimum of the 8: A 1.0106, B 0.9742. Mean of the 8: A 1.0222, B 1.0311. Median of the 8: A 1.0210, B 0.9970. The mean ranks the two the wrong way round on these 8 runs only if a spike hit B and not A; the minimum stays within the jitter of the floor.
<!-- /worked -->

The mean of these 8 ranks A first (1.0222 against 1.0311), because a spike hit B and not A. The minimum ranks them correctly and sits within the jitter of the floor, and the median is also right. This is one sample, and four procedures compared on 2,000 samples show what it stands for. Each procedure sees the same 20 launches of each candidate. The fourth splits them into 5 batches of 4, takes the minimum of each batch, and then treats the 5 batch minima as a sample of five independent numbers: their mean is the estimate and its standard error (Volume I, Chapter 12) is $s/\sqrt{5}$. The challenger displaces the incumbent only if the difference of the two estimates is more than twice the standard error of the difference, $2\sqrt{\mathrm{SE}_A^2 + \mathrm{SE}_B^2}$:

<!-- worked: kernel_selection.estimator_simulation -->
2,000 trials per row. In a trial, the incumbent A (floor 1) and the challenger B (floor = the ratio) each run 20 launches, A's first and then B's, 3 uniforms each (the noise of `kernel_selection.noise_runs`). The four procedures see the same 20 launches of each: 'mean' picks B when its mean is lower; 'median' when its median is lower; 'minimum' when its minimum is lower; 'batch-minimum, 2 SE' splits the 20 launches in 5 batches of 4, takes the minimum of each batch, and picks B only when the difference of the two means of the 5 batch minima exceeds twice sqrt(SE_A^2 + SE_B^2), SE = sd / sqrt(5) with divisor 4. The entries are the share of trials in which B was chosen. Generator: the book's SplitMix64, seed 0x383d4a69b75c684d (`seed_of` of the name `kernel_selection.estimators`, defined on the pseudocode page).

| floor of B / floor of A | mean | median | minimum | batch-minimum, 2 SE |
|---|---:|---:|---:|---:|
| 1.00 | 0.510 | 0.524 | 0.501 | 0.034 |
| 0.99 | 0.605 | 0.921 | 0.995 | 0.544 |
| 0.97 | 0.771 | 1.000 | 1.000 | 0.994 |
| 0.90 | 0.983 | 1.000 | 1.000 | 1.000 |
<!-- /worked -->

Three readings. The mean is the worst estimator even when B is 3 percent better: it chooses B in 77.1 percent of the trials, because one spike moves a mean of 20 by more than the difference. The median and the minimum find a 3 percent difference every time and a 1 percent one in 92 and 99.5 percent of the trials, but when the two candidates are identical they choose the challenger in about half the trials. That is a coin, and a coin written into a cache means that two runs of the same program on the same device can run different kernels; for identical candidates the harm is nothing, but for candidates that differ in accuracy or in a regime outside the measured shape it is a source of irreproducible behaviour. The rule of the fourth column keeps the incumbent in 96.6 percent of the identical cases and still finds the 3 percent difference 99.4 percent of the time. The price is stated in the 0.99 row, 54.4 percent: a 1 percent difference is inside the noise of 20 launches and the rule declines to act on it. A difference that small is not worth a cache entry that can flip. brain's tuner keeps an earlier candidate through a tie of 3 percent for the same reason, a fixed factor in place of the standard error of this chapter.

Two conventions bound the work. First, the measurement is a **race**: the incumbent gets its 20 launches, and each challenger is stopped when a launch has cost more than twice the best estimate so far; a challenger whose first batch of 4 is entirely stopped is dropped. Exactly: the cap of a challenger is twice the best candidate's mean of batch minima at the moment the challenger starts; a launch that would cost the cap or more is abandoned at the cap and charged the cap (the work counted in the tables is the sum of these charges); a challenger that survives its first batch runs all five batches. A candidate that is twice as expensive cannot win, and the cap spends at most $4 \times 2$ times the incumbent's cost finding that out instead of 20 launches of its own. A good candidate is dropped by a spike only if four of its launches in a row exceed twice its floor, which has a probability of at most $6.2 \times 10^{-10}$ in this noise model, for a challenger no worse than the best so far (Exercise 11). In the pipeline's first request on the GPU unit, tiled and naive are dropped after 4 launches each and the race takes 28 launches where 60 would be needed without it. Second, the conditions of the measurement are those of the use: the device is warmed up (the lesson of the cold roof), and a kernel that streams weights is timed on weights that have to come from memory and not from the cache, because a kernel that wins from the cache and loses from memory would be chosen for a situation it will not meet.

## The selection cache: keeping the answer where it holds

Measuring is device work, so the answer is worth keeping, but only where it is still true. What the answer is keyed by decides where it is reused and where it is discarded, and the work of the cache can be counted against what it saves.

A **selection cache** maps a key to the candidate chosen for it, and it is kept between runs. Its key has four parts, and each is a way the stored answer can stop being true:

- the **device identity**, because another physical device is another machine (two cards of one model, on one host, have different silicon and clocks, and a record for one is not a record for the other);
- the **driver and runtime version**, because the compiler that turns the same source into machine code is part of the kernel, and a new one can change which candidate wins;
- the **digest of the sources** of the candidates in the slot, because a kernel edited under the same name is another kernel;
- the **shape bucket**, because the winner is a property of a regime of shapes and one answer is kept for the whole of it.

The digest used here is FNV-1a over the bytes of the candidates' sources, 64 bits, the function brain uses for the same purpose: start with the 64-bit value $h$ = `0xcbf29ce484222325`, and for each byte $b$ of the text, in order, replace $h$ by $(h\ \mathrm{xor}\ b) \times$ `0x100000001b3` taken modulo $2^{64}$. It is not a cryptographic hash, and it need not be: an accidental collision between two versions of a kernel source is the only event it must prevent. The bucket of the rows is the smallest power of two not below $m$; $n$ and $k$ are exact, since a model has few distinct weight shapes.

<!-- worked: kernel_selection.cache_keys -->
FNV-1a over bytes, 64 bits: h = 0xcbf29ce484222325; for each byte b: h = h xor b, then h = h x 0x100000001b3 (mod 2^64). Test values: FNV-1a("") = 0xcbf29ce484222325, FNV-1a("a") = 0xaf63dc4c8601ec8c, FNV-1a("foobar") = 0x85944171f73967e8.

The stored entry has the key `gpu-unit7|drv-535.1|fd8d729ba11fb6bf|M<=128,N=2560,K=2048` and the value reg. The source text is `kernel reg: tile 128x128, block 8x8, step 8, barriers 2, guard rows yes, guard cols yes` (its edited form changes `step 8` to `step 4`). A request is looked up under its own key:

| request | key | lookup |
|---|---|---:|
| the same request | gpu-unit7|drv-535.1|fd8d729ba11fb6bf|M<=128,N=2560,K=2048 | hit |
| m = 100 (same bucket 128) | gpu-unit7|drv-535.1|fd8d729ba11fb6bf|M<=128,N=2560,K=2048 | hit |
| m = 129 (bucket 256) | gpu-unit7|drv-535.1|fd8d729ba11fb6bf|M<=256,N=2560,K=2048 | miss |
| another device | gpu-unit8|drv-535.1|fd8d729ba11fb6bf|M<=128,N=2560,K=2048 | miss |
| a driver update | gpu-unit7|drv-535.2|fd8d729ba11fb6bf|M<=128,N=2560,K=2048 | miss |
| an edited kernel source | gpu-unit7|drv-535.1|324f26863e720443|M<=128,N=2560,K=2048 | miss |
| n = 2561 | gpu-unit7|drv-535.1|fd8d729ba11fb6bf|M<=128,N=2561,K=2048 | miss |
<!-- /worked -->

A request with $m = 100$ falls in the same bucket as the stored $m = 77$ and hits; $m = 129$ is in the next bucket and misses; every other row changes one part of the key and misses. The rule is that a lookup that finds nothing measures, and an entry that cannot be read back is treated as absent and never trusted. brain also ignores a stored winner that is not among today's candidates.

**Bucketing is a trade between measurement work and precision.** Measuring every exact $m$ gives the best answer for each and costs a measurement for every new $m$; one bucket for all $m$ costs one measurement and serves most shapes with the wrong kernel. A workload of 400 requests with $m = \lceil 2^{9u} \rceil$ for a uniform $u$ (equal weight on every doubling), $n = 1024$ and $k = 2048$, under four schemes. A bucket is measured when a request first falls in it, at its largest member, with the race of the previous section:

<!-- worked: kernel_selection.bucket_tradeoff -->
A workload of 400 requests with n = 1024 and k = 2048, m = ceil(2^(9 u)) for a uniform draw u per request (so m runs from 1 to 512 with equal weight on every doubling). Generator: the book's SplitMix64, seed 0xc2f487ee2a7af15b (`seed_of` of the name `kernel_selection.workload`, defined on the pseudocode page). Draw order: one uniform per request, in order. A bucket is measured the first time a request falls in it, at its largest member (the powers of two: 1, 2, 4, ..., 512), with the chapter's protocol (20 launches of each of the three correct candidates, a 2 SE rule over the model's preference order). Measurement draws: `kernel_selection.buckets.<scheme>`, candidates in the model's preference order, 20 launches of 3 uniforms each. Costs are in billions (10^9) of equivalent bytes on the model GPU unit. 'Excess' is the work of serving the 400 requests with the buckets' choices minus the work with the best candidate for each exact m (16.28); the work of serving one request is the noise-free device cost B_dev of the chosen candidate at the request's own m, and the best candidate for an m is the one of least noise-free device cost (naive, tiled or reg; a measurement may pick another, and that is what the excess counts). 'Measurement work' is the sum of the costs of the launches of the races, each launch charged its noisy cost, a capped launch its cap. Measurement seeds, one generator per scheme and one race per bucket in the order of first request: one bucket per m: 0x3054a476c4518cf7; powers of two: 0xf807ce28e418167e; three buckets: 32, 128, 512: 0x7a1277ddb18f20cd; one bucket: 512: 0x04a8846b30f41c84. The race is the one of the text and of `kernel_selection.pipeline_trace`: the first candidate of the model's preference order (by predicted cost at the bucket's representative) is the incumbent and gets 20 launches in 5 batches of 4; each other candidate gets up to 20 launches, each capped at twice the incumbent's mean of batch minima, and displaces the best so far when its mean of batch minima is lower by more than 2 sqrt(SE_best^2 + SE_c^2) (SE = sd / sqrt(5), sd with divisor 4); a candidate whose first batch of 4 is entirely capped is dropped.

| scheme | buckets measured | measurement work | excess over the best, % | excess work |
|---|---:|---:|---:|---:|
| one bucket per m | 174 | 511.70 | 0.00 | 0.00 |
| powers of two | 9 | 22.62 | 2.82 | 0.46 |
| three buckets: 32, 128, 512 | 3 | 8.68 | 13.84 | 2.25 |
| one bucket: 512 | 1 | 2.73 | 64.32 | 10.47 |

Always using reg (the model's pick) costs an excess of 10.47, 64.32 percent.
<!-- /worked -->

The powers of two measure 9 buckets instead of 174 and spend 22.62 against 511.70 billion equivalent bytes of measurement, for an excess of 2.82 percent over choosing perfectly at each $m$. The error comes from the bucket 65 to 128, measured at 128, where the crossing at $n = 1024$ is at 113: the requests with $m$ between 65 and 112 run the register-blocked kernel where the tiled one was better. Three coarse buckets cost 8.68 and lose 13.84 percent. One bucket at 512 picks the register-blocked kernel for every $m$, which is exactly what the model said, and costs 64.32 percent more than the best. The ladder is geometric because the effects that matter (tile counts, padding, the fill of the lanes) change with the ratio of $m$ to the tile, not with its difference.

**The measurement is paid once and repays over runs.** The same workload, the cache kept between runs:

<!-- worked: kernel_selection.amortization -->
Same workload and measurements as `kernel_selection.bucket_tradeoff`. 'Saved per run' is the work of one run of the 400 requests under always-reg minus the work under the scheme's choices; 'runs to break even' is the measurement work divided by it. The last three columns are the measurement's share of all the work (requests plus the one-time measurement) after 1, 10 and 100 runs of the workload, the cache being kept between runs. Billions of equivalent bytes.

| scheme | measurement work | saved per run | runs to break even | share after 1 run, % | after 10 runs, % | after 100 runs, % |
|---|---:|---:|---:|---:|---:|---:|
| one bucket per m | 511.70 | 10.47 | 48.88 | 96.92 | 75.87 | 23.92 |
| powers of two | 22.62 | 10.01 | 2.26 | 57.48 | 11.91 | 1.33 |
| three buckets: 32, 128, 512 | 8.68 | 8.22 | 1.06 | 31.90 | 4.47 | 0.47 |
| one bucket: 512 | 2.73 | 0.00 | never | 9.26 | 1.01 | 0.10 |

The best possible saving per run (the exact-m choice) is 10.47.
<!-- /worked -->

With powers of two the measurement is repaid after 2.26 runs of the 400 requests (a run being one execution of the workload); measuring every $m$ takes 48.88 runs to repay and, after one run, is 96.9 percent of all the work. After 100 runs the measurement is 1.33 percent of the work of the powers of two. This is the argument for persisting the table (a process that restarts does not pay again), for a budget on the tuning (brain's tuner stops after a time budget and treats the classes it has not reached as unmeasured), and for a coarse ladder where the shapes are many.

The whole pipeline in one trace: four requests, the first two on the GPU unit around an edit of the register-blocked kernel's source (which changes its digest), the last on the CPU unit.

<!-- worked: kernel_selection.pipeline_trace -->
Catalogue, in the model's preference order: reg, tiled, naive, reg-notail (the register-blocked kernel without its row guard); naive is the reference. Request shape m x 1024 x 2048, rows bucketed to powers of two (m = 200 and 230 both fall in the bucket 256, which is measured at its largest member, 256). The gate checks each candidate except the reference on the bucket's two extreme shapes, 256 and 129, with the sampled positions of `kernel_selection.gate_table` (n = 1024 here), and charges the device one launch at each; the measurement is the race of the text (20 launches for the incumbent, challengers capped at twice the best estimate and dropped when their first batch of 4 is all abandoned). A launch is abandoned when its noisy cost reaches the cap of 2 x (the mean of batch minima of the best candidate so far, fixed when the challenger starts) and is charged the cap; every other launch is charged its noisy cost. The incumbent's 20 launches are 5 batches of 4 uncapped launches; a challenger that is not dropped runs all 5 batches. A candidate displaces the best when its mean of batch minima is lower by more than 2 sqrt(SE_best^2 + SE_c^2), SE = sd / sqrt(5) with sd taken with divisor 4. A gate launch is not noisy: it is charged the candidate's noise-free device cost at the gated shape and draws nothing from the generator for its cost. A gate shape's inputs are drawn as in `kernel_selection.gate_table`, afresh for each candidate and each shape; the gate stops at the first shape a candidate fails (the second is then neither drawn nor run, but both launches are counted and charged), reg and tiled are checked with the left-fold arithmetic of reg (k_rd = k = 2048), reg-notail with reg's arithmetic and the guard removed. The work of a miss is the gate launches plus the race. A request with one candidate left after the filter is not gated or measured. The key is `device|driver|digest of reg's source|bucket,n,k`. Generator: the book's SplitMix64, seed 0x53743f125dc83e63 (`seed_of` of the name `kernel_selection.pipeline`, defined on the pseudocode page). Draw order: for each miss, the gate's entries for each candidate that passed the filter and is not the reference (shape 256 then 129; the sampled rows of A, then the sampled columns of B), then the race's launches in the model's preference order (3 uniforms per launch). Work is in billions of equivalent bytes (gate launches plus measurement launches).

| step | m | after the capability filter | after the gate | gate launches | measurement launches | work | choice | cache |
|---|---:|---:|---:|---:|---:|---:|---|---|
| 1. first request, GPU unit | 200 | 4 | 3 | 6 | 28 (dropped: tiled, naive) | 3.00 | reg | miss |
| 2. same bucket | 230 | - | - | - | - | - | reg | hit |
| 3. after editing the source of reg | 230 | 4 | 3 | 6 | 28 (dropped: tiled, naive) | 3.06 | reg | miss |
| 4. the first request on the CPU unit | 200 | 1 | 1 | 0 | 0 | 0.00 | naive | miss |
<!-- /worked -->

The first request filters 4 candidates to 4, gates 3 (the reference is not gated, and `reg-notail` is refused at 129 after passing at 256), races 3, stores `reg`, and has spent 3.00 billion equivalent bytes. The second, $m = 230$ in the same bucket, costs nothing. The third, after the edit, is the first again: new digest, new key, a new measurement of 3.06. The fourth finds one admissible candidate and returns it without a gate or a measurement, since there is nothing to choose between. The selection as one function:

The helpers `select_kernel` calls are these. `digest` and `bucket_text` build the key; `order_by_predicted_cost` is the model's list; `gate_at` is one run of the gate at one shape; `race` is the measurement of the previous section, with `launch` returning the noisy cost of one launch of a candidate (its noise-free device cost $B_{\mathrm{dev}}$ times the factor of the noise formula, three uniforms $u_0, u_1, u_2$ in that order):

```text
function digest(catalogue: list<Candidate>) -> text:
    # FNV-1a (64 bits) of the sources in catalogue order, written as 16 lowercase hexadecimal digits
    return hex16(fnv1a64(utf8(c.source for c in catalogue, concatenated)))

function bucket_text(b: int, n: int, k: int) -> text:
    return "M<=" + str(b) + ",N=" + str(n) + ",K=" + str(k)

function order_by_predicted_cost(catalogue: list<Candidate>, m: int, n: int, k: int) -> list<Candidate>:
    # ascending B_pred(c) at (m, n, k); candidates of equal cost keep the order of the catalogue
    return stable sort of catalogue by B_pred(c, m, n, k)

function gate_at(c: Candidate, m: int, n: int, k: int, rng: Generator) -> bool:
    # one launch of c at the shape; the inputs are drawn as in the gate table (the entries of the sampled
    # rows of A, then of the sampled columns of B, each 2 * rng.uniform() - 1 rounded to f32)
    a, b = seeded inputs of shape [m, k] and [k, n] restricted to the sampled rows and columns
    return gate(value of c's output at each sampled position, a, b, k_rd(c, k))

function batch_minima(c: Candidate, m: int, n: int, k: int, cap: real, batches: int, rng: Generator) -> list<real>:
    out = []
    for j in 0 .. batches:
        best = infinity
        for i in 0 .. 4:
            cost = min(launch(c, m, n, k, rng), cap)       # a capped launch is charged the cap
            best = min(best, cost)
        append best to out
    return out

function race(cands: list<Candidate>, m: int, n: int, k: int, rng: Generator) -> Candidate:
    # cands is in preference order: cands[0] is the incumbent, with 20 launches in 5 batches of 4 and no cap
    v = batch_minima(cands[0], m, n, k, infinity, 5, rng)
    best, mu, se = cands[0], mean(v), sd(v) / sqrt(5)
    for c in cands[1:]:
        cap = 2 * mu                                         # twice the best estimate when c starts
        first = batch_minima(c, m, n, k, cap, 1, rng)
        if first[0] >= cap:
            continue                                         # the whole first batch was abandoned: dropped
        v = first + batch_minima(c, m, n, k, cap, 4, rng)
        if mu - mean(v) > 2 * sqrt(se^2 + (sd(v) / sqrt(5))^2):
            best, mu, se = c, mean(v), sd(v) / sqrt(5)
    return best
```

```text
function select_kernel(m: int, n: int, k: int, catalogue: list<Candidate>, reference: Candidate, caps: Caps,
                       device_id: text, driver: text, cache: inout list<(text, text)>, rng: Generator) -> text:
    b = 1
    while b < m:
        b = 2 * b                                   # the bucket: the smallest power of two not below m
    key = device_id + "|" + driver + "|" + digest(catalogue) + "|" + bucket_text(b, n, k)
    for (stored_key, name) in cache:
        if stored_key == key and any(name == c.name for c in catalogue):
            return name                             # a stored winner that is not a candidate today is ignored
    cands = [reference]
    for c in order_by_predicted_cost(catalogue, b, n, k):      # best guess first
        if c.name != reference.name and admissible(c, caps, b, n, k):
            append c to cands                       # stage 1: capability filter, a correctness gate
    if len(cands) == 1:
        append (key, reference.name) to cache
        return reference.name
    keep = []
    for c in cands:
        if c.name == reference.name or (gate_at(c, b, n, k, rng) and gate_at(c, b // 2 + 1, n, k, rng)):
            append c to keep                        # stage 2: reference gate, the bucket's two extremes
    choice = race(keep, b, n, k, rng)               # stage 3: 20 launches, 2 SE rule, cap at twice the best
    append (key, choice.name) to cache              # stage 4: remember with the key
    return choice.name
```

Here the digest is the FNV-1a of the concatenated sources as above (so a different source set gives a different key; in the worked tables the only source that changes is that of `reg`, and the digest printed there is that of the text of `reg` alone), and the helpers are the functions above. `fnv1a64` is the function of the cache section, `utf8` the byte string of a text, `hex16` writes a 64-bit value as 16 lowercase hexadecimal digits, and `k_rd(c, k)` is $k$ for a left fold and $1 + \lceil \log_2 k \rceil$ for a tree. A candidate is gated at the bucket's largest member $b$ and at its smallest, $b/2 + 1$, and the second gate is not run when the first fails.

## Selection in brain: the selector, the roofs and the tuning tables

**The seam: `crates/backend-api/src/select.rs`.** The module header names what it replaced: every call site wrote its own regime test (`if m <= 32 && gpu.kind() != "cpu"`), which scattered the policy, made it untestable without a device and keyed it on a backend's name rather than on what the device can do. An `Op` and an `OpShape` (m, n, k and a data type) name the operation and the shape; a `KernelVariant` (`Reference`, `WorkgroupPerOutput`, `SplitReduction`, `PackedInt8`, `RegisterTiled`, `FusedFlash`) names the implementation family, and the call site maps a family to its own pipeline index, with `Reference` as the fallback every model ships. `candidates` returns the variants that can execute for an op, a shape and a device, static best guess first: a match enumerates the regimes (the order is "measured regime boundaries", no capability checks), and a single uniform filter at the end keeps what the device's capabilities permit and returns `Reference` if nothing is left, so that the list is never empty. The default policy is the head of that list by construction (`DefaultSelector`), "so the default choice and the tuner's probe set can never drift apart", and a test pins it (`candidates_head_is_the_default_policy`). The matrix-product arm is the problem of this chapter in code: for fp32 and the storage types it enumerates the decode kernel, the register-tiled kernel from `GEMM_TILE_MIN_ROWS` = 8 rows and `GEMM_TILE_MIN_COLS` = 128 columns, and the reference, thresholds the comment says were swept at $k = 2048$, $n = 2560$ (knowledge #15 and commit 6cfc15861 are that sweep).

**Capability declarations.** `Requirement` in the same file is the declaration of what a variant needs, as data: flags for packed-int8 dot products, fast f16 and bf16 compute, storable f16, bf16 and fp8, workgroup reductions, and a matrix-engine requirement (a triple of types that `satisfied_by` matches against the shapes the device enumerated). Every field defaults to no constraint, so an empty requirement is always satisfied, and `KernelVariant::requires` is the one table, filtered in one place. The doc of `workgroup_reductions` says what the chapter's filter says: "so this is a correctness gate, not a preference", because the CPU compiler's split-at-barrier execution mis-executes such kernels (Chapter 9; knowledge #26). The half-precision field says the other half: availability of the extension is "never treated as a speed claim on its own", since a device can expose it at a small fraction of its fp32 rate (`crates/gpu-core/src/roof.rs` records Pascal). The test `no_candidate_ever_requires_an_unsupported_capability` runs every op against every data type. Where a candidate is native CUDA, `crates/kernels-cuda/src/lib.rs` carries the declaration in `CudaKernel`: `min_cc` (the lowest compute capability its text is valid on, "the instructions it uses, not the cards it was tried on"), the output `tile`, `block_dim`, `shared_bytes`. `check_table` is the check of a declaration against its source that the chapter asked for: a text with `__dp4a` cannot declare a floor below compute capability 6.1, `mma.sync` or `cp.async` below 8.0 (unless the source gates them itself), `wgmma.mma_async` below 9.0, and a kernel may not declare more shared memory per block than every capability guarantees ("a card that grants more must be QUERIED, never assumed"). `best_for` picks the entry with the highest floor not above the device's, a static preference for the newest instruction set; where several candidates share a slot the measured table below replaces it.

**The measured roofline: `crates/gpu-core/src/roof.rs`.** The compute probe is `crates/kernels/wgsl/roof_fma.wgsl` (eight accumulators, `c` and `d` through the uniform, a fixed point of $c = d = 0.5$ so there is no overflow, no denormal and no not-a-number, and an explicit `fma`); the bandwidth probe is `axpy` over two buffers of $2^{26}$ floats, 12 bytes per element, which is the arithmetic of the section above. Both self-calibrate their trip count until the timed region is long enough, bracket each dispatch with a wait for the device, bound each wait, and run after a warm-up of continuous work. The result, `Roofs`, adds the ridge, a cache-resident bandwidth, and optional rates for packed int8 and native f16; `utilisation` grades work against the roof it belongs to and `Bound::target_pct` and `defect_pct` give the bands (60 and 30 percent for compute-bound work, 70 and 35 for memory-bound). Roofs are cached per backend and physical device and persisted under a name that carries the adapter, the backend, the card's identity and a hash of the probe sources. Three stories are in its history. Commit 73466fee0 (August 2026) replaced the hard-coded peaks that every utilisation figure divided by (knowledge #27).

<!-- perf-number: historical measurement, commit 73466fee0 -->
In August 2026, on one Tesla P40, the commit reports measured rates of 10,555 of 11,760 (GFLOP/s) and 287.6 of 346 (GB/s) of the specification, 89.8 and 83.1 percent.

Commit 1bbcfc77e (August 2026) found the cold probe of this chapter: on an integrated GPU the probe started timing at the idle clock, recorded it as the silicon's ceiling, and the file kept it, so that the fp32 roof came out below the rate of a real matrix product measured on the same device; the commit added the warm-up and a test (`real_work_can_never_beat_the_roof` in `crates/gpu-core/tests/it/roofline.rs`) for the invariant. And knowledge #213 records that the compute probe had been written $a \leftarrow a c + d$, which a backend that forbids contraction compiles to two instructions, so the roof it reported was half the fused rate and a kernel issuing explicit fused multiply-adds appeared to beat the silicon; the probe now says `fma`, and the header of the kernel says why.

**The tuner and the stores.** `AutoTuner` in `select.rs` is the measuring stage in its general form: `resolve` takes the op, the shape, the device's capabilities and a closure that returns a cost for a candidate, consults its memo, then the persistent store, then measures each candidate of `candidates` once, and remembers and persists the winner; a measurement that fails (`None`) removes the candidate, and with autotuning off (`--no-autotune`, read once at construction) it answers with the static policy, which is what a reproducible benchmark needs. A persisted value that is not among today's candidates is ignored ("the stale-cache rule"). The store is a trait; `FileTuneStore` in `crates/gpu-core/src/tune.rs` keys its file by the adapter and a fingerprint of the candidate kernel sources, FNV-1a, the function of the cache section, so that "edit a candidate kernel and the old winners stop applying by filename", merges concurrent writers per key and renames atomically, and ignores a file it cannot parse. `resolve` takes the strict minimum with no margin and does not itself check outputs; the checking is the tests' and the callers'. `gemm_schedule_candidates` bounds a schedule search (a split of $k$ across workgroups) to at most three candidates with the static guess first; commit 52ec5df99 (September 2026) added it. Commit 3d0bcb3b2 (September 2026) added a $64 \times 64$ tile as another schedule, measured it on real hardware with interleaved runs, and reports a clear loss at an $m$ aligned to 128 and, at the ragged shape it was meant to help, a result "indistinguishable from this box's own documented thermal noise (direction flipped across six trials)". It reverted the live search to its earlier behaviour and kept the kernel and its correctness gate, a decision the rule of the noise section justifies: an improvement that flips direction between repeats is not an improvement.

**The measured tables and the admission rule: `crates/gpu-core/src/native_tune.rs`.** For the int8 decode linear, each slot lists several native candidates (the dp4a GEMV, a wide tensor-core GEMV, 16- and 32-row tiled tensor-core GEMMs and others, in `crates/gpu-core/src/native_upgrade.rs`, each with the compute capability it needs, the shapes it `serves`, the alignment of its bindings it needs, and the form of its inputs); commit 72b363135 (October 2026) introduced the mechanism and knowledge #368 records what the first run showed. On the first handle of a device a grid of shape classes (row bucket by $K$ class by weight-byte class) is built; every candidate that serves a class is run on a synthetic problem of it, with the output buffer filled with not-a-number first, and its error against an f64 reference is compared with the admission rule; the survivors are timed by the backend's per-launch events, the minimum over several short runs of launches, with the weights cycled past the L2 where the class streams them; an earlier candidate keeps a tie inside a factor `TIE` of 0.97; and the choice is stored with its achieved weight bandwidth and its fraction of the measured roofline. `report` and `brain perf roofline gpu --kernels` print the table and the classes under 60 percent of their bound. `Tuned::choice` looks the class up per dispatch in constant time.

The admission rule, `admit`, is the chapter's gate with the tolerance derived from formats instead of from $k_{\mathrm{rd}}$ and $S$: each candidate declares the form of its inputs (`Input`: int8 activations with a scale per 32, bf16 rows, E4M3 with a scale per 128), the tuner computes in f64 the error of rounding the inputs to those forms, and a candidate is admitted when its own error is at most `ALPHA` = 8 times that error, both as root mean square and as maximum over the sampled outputs, never below the fp32 accumulation floor. Knowledge #376 records why: a fixed bar written for one arithmetic (`2e-5 * rms`, against the int8 semantics) rejected a bf16 tensor-core candidate whose error was the rounding of its own inputs, a hundred times the bar, so a fixed bar against one arithmetic admits no other. The line follows from the declared formats and needs no per-candidate tolerance. The sampled positions are those of the gate above (`sample_positions`: first, middle and last rows, columns at tile edges).

**Where the limits show.** The gate runs on the synthetic problem of a class, not on every shape in it. The test `every_prefill_candidate_agrees_with_the_f64_oracle_on_ragged_shapes` in `crates/gpu-core/tests/cuda_provider_matmul_i8.rs` says why that is not enough: "a candidate that is wrong on a shape the tuner never timed is a defect waiting for the next model", so every candidate is forced in turn through the production chain (`force_candidate`) and checked on shapes with partial tiles, odd widths and bindings at every alignment (`every_candidate_of_the_slot_agrees_with_the_wgsl_reference` in `crates/gpu-core/tests/it/i8_gemv_native.rs` does the same for the decode slot). The persisted native table is keyed by the device description, the backend, the compute capability, the candidates' sources, the crate version, the admission constant and the grid, so that a changed rule retunes (`device_key`); the key as coded has no driver-version field, so a driver update that changes which candidate wins does not by itself retire a stored table (Exercise 10). The Vulkan pipeline cache of Chapter 9, by contrast, is keyed by the driver's cache identifier.

**Two stories of a selection that failed quietly.** Commit d10c09576 (July 2026) introduced `AutoTuner`, `candidates` and the file store, and in wiring `candidates` into the serving engine it exposed a regression that the commit says was silent: the gate for int8 weights probed the head of the selector for the packed-int8 variant at one row, where the head had become the GEMV, so int8 fell back to fp32 and the test that compared the two skipped instead of failing. The code now gates on the capability itself (`let w8_on = weights_int8 && caps.numeric.int8_dot;` in `crates/qwen3/src/serve.rs`), and a test asserts that a device that reports the capability cannot reach the skip. It is the failure of the chapter's first stage turned inside out: the question "may this variant run" was asked of a preference. The second is commit a1ec87194 (October 2026): the tuner has a time budget; classes past it were saved with no results and their default candidate, and the loader accepted such a file as complete, so for the life of the build every class it never reached ran on the single-row GEMV. A stored table with unmeasured classes is now incomplete, the next start keeps the measured classes and times only the rest (tests `a_stored_table_with_unmeasured_classes_is_incomplete` and `a_class_the_budget_did_not_reach_takes_its_nearest_measured_neighbours_choice`), and until then an unmeasured class takes the choice of the nearest measured row bucket whose winner serves it. 

<!-- perf-number: historical measurement, commit a1ec87194 -->
In October 2026, for one decoding step of a 27B model with 32 sequences at 32K context on one GH200, the commit reports 76.9 ms before the retune against 47.4 ms after, the linears running at a third of their weight bandwidth while the table was incomplete.

A cache entry that merely records the absence of an answer is a wrong answer with a valid key. Knowledge #105 and #132 are the neighbours of both: a selector can be right and still select between kernels that are not interchangeable (a constant hard-coded in one of two kernels a capability picks between), and a by-name selection seam that falls back to the slow kernel when the fast one is not registered fails open and says nothing.

## Check your understanding

1. A roofline model ranks candidate A above candidate B at every shape of a sweep, and the device ranks B above A at small $m$. A colleague concludes that the model's FLOP count must be wrong. Find the error and say what the model's numbers are good for.
2. Find the error: "On the CPU unit the register-blocked kernel has the best predicted cost, so we rank it first and let the tuner discover that it does not run."
3. A device declares a half-precision unit that runs at a small fraction of its fp32 rate. Should the capability filter admit a half-precision candidate for the fp32 contract? Should the selector pick it?
4. `reg-notail` passes the gate at $m = 256$ and the pipeline still refuses it. Which other shape makes it fail, and what does this say about the relation between the cache key and the gate?
5. Find the error: "We measured the compute roof by running our best matrix multiplication, so that every kernel is graded against the best we have."
6. After a graphics-driver update the cache lookup hits and the engine runs the old choice. Which part of the key was missing, what could have changed, and what is the cost of adding it?
7. The challenger's estimate is 1.5 percent below the incumbent's, and its standard error of the difference is 1 percent. Which does the rule of this chapter choose, and what would a plain minimum choose?
8. Why is the bucket ladder geometric, and what goes wrong with a representative taken from the middle of a bucket instead of its largest member?

## Summary

The roofline model of Chapter 3 ranks implementations by their own bytes and FLOPs, and the ranking is wrong exactly where a kernel is far above its bound: on the model GPU unit it is wrong on 8 of 16 shapes, by up to 62.5 percent of what was available, because the model sees neither the idle lanes of a tile, nor the invocations that exist, nor the instruction mix, nor the start of the launch. It remains a lower bound, an order for the candidates and the source of the floor in which results are stated. The choice itself is made on the device, by a pipeline of stages each of which prevents one failure. An **implementation candidate** states its conditions, and a **capability declaration** against the device's record removes what cannot run before any cost is computed, as a correctness gate and never a preference. The **measured roofline** of two probes whose work is counted (166.9 (GFLOP) and 2.4 (GB) for the GPU unit) gives the denominator of every fraction, and a fraction above 1 beyond the noise signals a bad roof. **Reference-gated selection** admits a candidate only after its output on the device matches the reference within a tolerance derived from its arithmetic, at the edges of the bucket as well as at its representative, because the fastest candidate is often the one that does less. A robust statistic, a rule that declines differences inside the noise and a race that drops hopeless challengers early make the stage that times candidates stable. The **selection cache**, keyed by device, driver, source digest and shape bucket, keeps the answer where it holds, and a geometric ladder of buckets repaid its measurement after 2.26 runs of the workload at a cost of 2.82 percent over the best possible choice.

What the chapter leaves open is how to know that a timing is a timing of the kernel: a clock stopped before the device finished, a run that includes a compile, a mean that hides a spike. Selection is only as good as its measurements, and the next chapter is about producing numbers that do not lie.

## References

- R. Clint Whaley and Jack Dongarra, "Automatically Tuned Linear Algebra Software", 1998.
- Matteo Frigo and Steven G. Johnson, "The Design and Implementation of FFTW3", 2005.
- Samuel Williams, Andrew Waterman and David Patterson, "Roofline: An Insightful Visual Performance Model for Multicore Architectures", 2009.

## Exercises

1. **[Calculate]** For $m = 77$, $n = 2560$, $k = 2048$, find $N_{\mathrm{byte}}$, $N_{\mathrm{flop}}^{\mathrm{pad}}$, $o_{\mathrm{cmp}}$, $o_{\mathrm{mem}}$, $B_{\mathrm{pred}}$ and $B_{\mathrm{dev}}$ of the tiled kernel from the formulas and the parameters of `kernel_selection.candidate_table` and `kernel_selection.cost_model_example`, and check them against the table.
2. **[Calculate]** For the register-blocked kernel at $m = 1024$, $n = 130$, $k = 2048$, find the tile grid, the invocations, $o_{\mathrm{cmp}}$ and $o_{\mathrm{mem}}$, and say which term of $B_{\mathrm{dev}}$ is the maximum. Compare with the last row of `kernel_selection.crossover_table`.
3. **[Calculate]** A compute probe of $2^{20}$ invocations with 16 FLOPs per step is accepted when its cost is at least 100 launches of 5 microseconds, on a device whose compute rate is 36,000 (GFLOP/s) (no noise). The cost of a run is the launch plus the work over the rate, as in `kernel_selection.calibration`. Find the smallest trip count whose cost is at least 100 launches, the FLOPs of the attempt and the FLOPs of its three runs. Do the same for the bandwidth probe, at 664 (GB/s) and 805,306,368 bytes per pass.
4. **[Calculate]** A left fold over $K = 4096$ products and a tree over the same products: find $k_{\mathrm{rd}}$ and the tolerance of each in units of $u S$. From which power of two $K$ is the tree's tolerance below 1 percent of the left fold's?
5. **[Calculate]** The FNV-1a hash of the two-byte text `ab`, by hand or with a program, with the constants of the chapter (a check: you should get `0x089c4407b545986a`; the test values for `a` are in `kernel_selection.cache_keys`). Then change one byte of the text and say how many of the 64 bits change.
6. **[Implement]** Reproduce `kernel_selection.ranking_table`: the three candidates, the counts, $B_{\mathrm{pred}}$ and $B_{\mathrm{dev}}$ as defined above, the noise factor from three uniforms of the book's generator with the seed and the draw order printed there, and the minimum of 8 runs. Add a fourth candidate, a $64 \times 64$ tile with a $4 \times 4$ block and 256 invocations per workgroup, with the loads in flight and the issue efficiency by the rule of `kernel_selection.candidate_table` (a $64 \times 8$ slab of A and an $8 \times 64$ slab of B over 256 invocations is 4 loads in flight; $e = 16/(16 + 4 + 4 + 2)$), and find on how many shapes of the table it is the device's pick.
7. **[Implement]** Reproduce `kernel_selection.calibration` for the CPU and the GPU unit from the probe loop of the text and the seeds printed there, and report the measured rates and the work of the probes. Add a bandwidth probe over a working set of 4 MiB and say what the measured rate is a rate of.
8. **[Implement]** Reproduce `kernel_selection.gate_table`: the float32 left fold and tree, the exact reference, the tolerance and the 27 positions, from the seed and the draw order printed there. Add a candidate that drops the last column tile when $n$ is not a multiple of 128 and find a shape of the bucket at which the positions of the chapter do not detect it.
9. **[Implement]** Reproduce `kernel_selection.noise_runs` and `kernel_selection.estimator_simulation`. Add a fifth procedure, the minimum of 20 launches with a margin of 3 percent as brain's tuner keeps its earlier candidate (the challenger B replaces the incumbent A only when the minimum of B's 20 launches is below 0.97 times the minimum of A's), and report its choice rates at the four ratios.
10. **[Investigate]** Read `device_key` in `crates/gpu-core/src/native_tune.rs` and `FileTuneStore` in `crates/gpu-core/src/tune.rs`. Say which parts of this chapter's key each store has, design a test that shows whether a driver change invalidates a stored table (what must the test fake, and what does it assert), and decide whether adding the driver to the key could cost more than it saves.
11. **[Investigate]** With the noise of the chapter, bound the probability that a good candidate is dropped by the cap: a launch exceeds twice the floor only if the spike $s$ satisfies $1.04(1 + s) > 2$. Compute the bound for one launch and for four, and then run the pipeline's race 10,000 times with a seeded generator and report how often the dropped candidate was the true best.
12. **[Investigate]** On a machine you have, time two implementations of a kernel of your own over a range of one size parameter, with the procedure of this chapter (a warm-up, 20 launches, batch minima, a 2 SE rule). Report the crossing, its spread over three repetitions and whether a fixed threshold of the form $m \ge c$ would have been right.

## Key terms

- **implementation candidate**: one implementation of an operation together with the conditions under which it is valid (the capabilities it needs, the shapes it serves, the contract of its result), one of several that a selector chooses among. <!-- concept: implementation-candidate -->
- **capability declaration**: the statement of what a device can execute correctly, queried from the device and absent when unknown, matched against what each candidate needs, so that a candidate that cannot run is removed before any cost is considered. <!-- concept: capability-declaration -->
- **measured roofline**: the peak arithmetic rate and the memory bandwidth of the device itself, found by two probes that do nothing but arithmetic and nothing but streaming and whose work is counted, in whose terms every result is stated as a fraction. <!-- concept: measured-roofline -->
- **reference-gated selection**: choosing among candidates only those whose output on the actual device matches a reference within a tolerance derived from their arithmetic, checked at the edges of the shapes they will serve, before any of them is timed. <!-- concept: reference-gated-selection -->
- **selection cache**: the stored map from a key of device identity, driver version, source digest and shape bucket to the candidate chosen, kept between runs so that a measurement is paid once and discarded when any part of the key changes. <!-- concept: selection-cache -->

## Answers

**Recall questions.**

1. Intensity $2 \times 10^9 / 10^8 = 20$ FLOP per byte, below the ridge point of 50, so bandwidth-bound. $B_{\mathrm{eq}} = \max(10^8,\ 2 \times 10^9 / 50) = 100{,}000{,}000$ bytes.
2. One row of 128: $1/128 = 0.8$ percent.
3. $\mathrm{BW} \cdot t_{\mathrm{lat}} = 800 \times 10^9 \times 200 \times 10^{-9} = 160{,}000$ bytes in flight, which is $160{,}000 / 4 = 40{,}000$ loads of 4 bytes. With fewer in flight the memory idles part of the time and the bandwidth is not reached. (This unit's 664 (GB/s) needs 33,200.)
4. The value that forbids the feature (zero, false, absent). A missing answer can then only make the engine more cautious: it is never assumed to be able.
5. $k_{\mathrm{rd}} = 8$, so $8 \cdot 2^{-24} \cdot 4 / (1 - 8 \cdot 2^{-24}) \approx 1.9 \times 10^{-6}$.
6. $0.01/\sqrt{5} = 0.0045$.

**Check your understanding**

1. The FLOP and byte counts can be exactly right and the ranking still wrong. The model is a lower bound for each candidate, $B_{\mathrm{dev}} \ge B_{\mathrm{pred}}$, and the gap differs: A's tile is mostly idle at small $m$, which the useful FLOPs and the requested bytes do not show. A ranking of lower bounds is not a ranking of costs. The numbers are good as the order of the candidate list, as the lower bound that no measurement may violate and as the floor of a fraction.
2. The register-blocked kernel cannot execute on the CPU unit (no workgroup barriers): there is no cost to rank and nothing to "discover", only a defective run. The filter comes first and no cost, modelled or measured, is computed for an inadmissible candidate.
3. Admit it: the declaration says that the feature executes correctly, and the contract is another question. A kernel for fp32 operands, with fp32 results at the tolerance of the slot, is the contract; a half-precision matrix product has a different one and belongs to a different slot. Whether a candidate of the right contract that uses the unit is chosen is decided by the measurement, not by the filter.
4. Any shape of the bucket that is not a whole number of 128-row tiles, for instance 129 or 200 (rows 128 and 199 are left unwritten). The cache entry covers the whole bucket and the gate covers only the shapes it was run at; the gate must cover every class of failure (a full tile, a partial one) of the shapes that map to one key.
5. A roof derived from the best kernel is the best kernel's rate, so that kernel is at 1.0 by definition and every other is graded against the engine's own best, which hides the gap to the silicon. The roof must come from probes that do only arithmetic and only streaming.
6. The driver (or runtime) version. A new compiler can generate different code for the same source, so the winner can change. Adding it costs a re-measurement after each update (a few runs of work, once), which is cheap against running a stale choice for the life of the install; the cost grows only if the driver string changes spuriously.
7. The rule keeps the incumbent: the gap, 1.5 percent, is below twice its standard error, 2 percent. A plain minimum would choose the challenger, which at a 1 percent true difference it does in 99.5 percent of the trials of the simulation and at identical candidates in about half of them.
8. The effects that decide the winner (tile counts, padding, fill of the lanes) depend on the ratio of $m$ to the tile, so a bucket that doubles at each step keeps the same relative width. A middle member is smaller than most of the bucket; its winner may be the tiled kernel while the bucket's larger shapes need the register-blocked one, and the gate and the measurement then describe shapes the bucket does not mostly contain.
