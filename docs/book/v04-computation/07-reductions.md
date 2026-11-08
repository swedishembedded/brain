<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Parallel Reductions, Scans and Sorting

*How do thousands of threads add up one array, and why is the obvious way slow?*

Sixteen numbers sit in memory, $x = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3]$, and a kernel has sixteen invocations, one for each. We want their sum. A loop would add them one after another. The machine of Chapter 4 offers sixteen workers instead, and the obvious program is the one the kernel model of Chapter 5 invites: every invocation does its own small piece, here "add my number to the total".

```text
kernel sum_one_cell(x: array<f32>[N], total: inout array<f32>[1], N: int) workgroup n_wg:
    if g < N:
        atomic_add(total, 0, x[g])
```

An **atomic update** (the pseudocode page defines `atomic_add`) adds into a word as one indivisible step, so that two invocations updating the same word cannot lose each other's addition. The program is correct. It is also a loop in disguise. All sixteen updates go to the same word, a word serves one update at a time, and so the sixteen "parallel" additions are served one per step in an order the device chooses:

<!-- worked: reductions.first_sixteen -->
Sixteen numbers x = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3], one per invocation, sum 80. A tree step with stride s makes invocation l < s add the value of invocation l + s to its own; the table lists the values the active invocations hold after each step (the others are not used again).

| step | stride s | invocations adding | values after the step | additions so far |
|---|---:|---:|---:|---:|
| 0 | - | - | [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3] | 0 |
| 1 | 8 | 0 .. 7 | [8, 4, 9, 9, 14, 16, 11, 9] | 8 |
| 2 | 4 | 0 .. 3 | [22, 20, 20, 18] | 12 |
| 3 | 2 | 0 .. 1 | [42, 38] | 14 |
| 4 | 1 | 0 .. 0 | [80] | 15 |

The other way: all 16 invocations update one cell that starts at 0. The cell serves one update per step, so after step j it holds the sum of the first j numbers in the order the hardware chose; in order 0 .. 15 it holds [3, 4, 8, 9, 14, 23, 25, 31, 36, 39, 44, 52, 61, 68, 77, 80] after steps 1 .. 16.

| way | dependent steps (depth) | additions (work) |
|---|---:|---:|
| one cell, every invocation updates it | 16 | 16 |
| tree | 4 | 15 |
<!-- /worked -->

The other way never lets two invocations touch the same word. In the first step the invocations $0 \ldots 7$ each add the value of the invocation eight places to the right to their own; in the second, $0 \ldots 3$ add the value four places away; and so on, halving. After four steps invocation 0 holds 80. It took 15 additions, one fewer than the cell's 16 updates, and four steps in a row instead of sixteen. A reduction carried out by many invocations at once is a **parallel reduction**, and this one, a **tree reduction**, is the idea of Chapter 16 (Volume I) made into a kernel. This chapter builds the kernel, finds what it costs and finds what it changes.

Two numbers describe any such computation, and both can be counted before running anything. The **work** $N_{\mathrm{work}}$ of a computation is the number of operations it performs. Its **depth** $N_{\mathrm{dep}}$ is the length of the longest chain of operations in which each one needs the result of the one before: the steps it would still take if there were as many workers as it could ever use. For $n$ numbers:

<!-- worked: reductions.work_depth -->
The three ways to sum n numbers, counted. Work = additions; depth = the longest chain of additions that wait for one another; steps = how long n_wk workers need, n_wk = min(n, 10,240) (the lanes of the model GPU of Chapter 4). For a computation of work N_work and depth N_dep the steps are at least max(ceil(N_work / n_wk), N_dep), and a greedy schedule needs at most ceil((N_work - N_dep) / n_wk) + N_dep; the tree row prints both. The cell serves one update per step whatever n_wk is.

| n | way | work | depth | steps on n_wk workers |
|---:|---|---:|---:|---:|
| 16 | one invocation walks the array | 15 | 15 | 15 |
| 16 | every invocation updates one cell | 16 | 16 | 16 |
| 16 | tree | 15 | 4 | 4 to 5 |
| 1,024 | one invocation walks the array | 1,023 | 1,023 | 1,023 |
| 1,024 | every invocation updates one cell | 1,024 | 1,024 | 1,024 |
| 1,024 | tree | 1,023 | 10 | 10 to 11 |
| 1,048,576 | one invocation walks the array | 1,048,575 | 1,048,575 | 1,048,575 |
| 1,048,576 | every invocation updates one cell | 1,048,576 | 1,048,576 | 1,048,576 |
| 1,048,576 | tree | 1,048,575 | 20 | 103 to 123 |
<!-- /worked -->

A loop and the single cell have depth $n - 1$ and $n$: no number of workers helps, because the chain is the whole computation. The tree has the same work, $n - 1$ additions, and a depth of only $\lceil \log_2 n \rceil$. With $n_{\mathrm{wk}}$ workers no schedule can finish in fewer than $\max(\lceil N_{\mathrm{work}} / n_{\mathrm{wk}} \rceil, N_{\mathrm{dep}})$ steps (the workers cannot do more than their share, and cannot beat the chain), and a greedy schedule that never leaves a worker idle when an operation is ready finishes in at most $\lceil (N_{\mathrm{work}} - N_{\mathrm{dep}}) / n_{\mathrm{wk}} \rceil + N_{\mathrm{dep}}$ steps (Brent's bound, 1974). For $2^{20}$ numbers on the 10,240 lanes of the model GPU those are 103 and 123 steps, against $2^{20}$ for the cell. At the size of a real tensor the depth is no longer the limit, the work divided by the workers is; the first rows of the table are the regime where depth rules.

So the question of the chapter has two halves. Why is the obvious way slow? Because its depth is its work: the shared word is a queue. And how do thousands of threads add up an array? With a tree that never makes two of them wait for the same word, spread over the machine's levels of cooperation: inside a workgroup with shared memory and barriers, across workgroups with several dispatches. The tree changes the order of the additions, which for floating-point numbers changes the answer, and the same pattern (a reduction, a prefix sum, a sort built from them) turns up wherever a kernel needs a total, an offset or an ordering. The chapter takes them in that order.

Six ideas from earlier chapters carry the argument, from the tree of Volume I to the barrier rule of Chapter 5. Try them from memory first; the answers are at the end of the chapter, under "Recall questions".

1. (Volume I, Chapter 16, reduction algorithm.) How many layers does a tree need to add 1,000 numbers, and how many additions does it perform?
2. (Volume I, Chapter 16, prefix sum.) Write the exclusive scan of $[2, 7, 1, 8]$. How many additions does the doubling scan of that chapter use for $n = 8$?
3. (Volume I, Chapter 15, floating point.) In float32, what are $(1 + 2^{-24}) + 2^{-24}$ and $1 + (2^{-24} + 2^{-24})$?
4. (Volume IV, Chapter 4, workgroup.) Two invocations belong to different workgroups of one dispatch. Can the first wait for something the second writes?
5. (Volume IV, Chapter 4, occupancy.) A kernel has workgroups of 64 invocations, 24 registers per invocation and 512 bytes of shared memory per workgroup. How many workgroups are resident on a compute unit of the model, and what is its occupancy?
6. (Volume IV, Chapter 5, the barrier rule.) A kernel begins with `if g >= N: return` and then reaches a `barrier`. When is that legal?

## Summing in one workgroup: the tree

A workgroup can share memory and wait at barriers, so inside one workgroup the tree can be written down exactly. We fix the kernel, check it by simulation and count what it uses. Let a workgroup of $n_{\mathrm{wg}}$ invocations (a power of two) add $n_{\mathrm{wg}}$ numbers. Each invocation first copies its number into the shared array `s`. Then comes a series of steps with stride $s = n_{\mathrm{wg}}/2, n_{\mathrm{wg}}/4, \ldots, 1$: in the step of stride $s$ the invocations $l < s$ add the slot $l + s$ to the slot $l$. A barrier follows each step.

```text
kernel reduce_tree(x: array<f32>[N], part: inout array<f32>[n_grp], N: int) workgroup n_wg:
    shared s: array<f32>[n_wg]
    if g < N:
        s[l] = x[g]
    else:
        s[l] = 0.0                      # the identity of the sum: it changes no total
    barrier
    stride = n_wg // 2
    while stride >= 1:
        if l < stride:
            s[l] = s[l] + s[l + stride]
        barrier
        stride = stride // 2
    if l == 0:
        part[w] = s[0]
```

The guard of the load writes a value in every case, so every slot is defined before the first barrier, and a workgroup that owns fewer than $n_{\mathrm{wg}}$ real numbers pads with 0, the **identity** of the sum (for a maximum it would be the smallest number). The `if l < stride` surrounds the addition and never a barrier, and the loop's condition is the same for every invocation, so every invocation reaches every barrier, as the rule of Chapter 5 demands. The simulator of that chapter runs this kernel on the sixteen numbers of the problem:

<!-- worked: reductions.tree_kernel_trace -->
The workgroup kernel `reduce_tree` of the chapter (sequential addressing) run by the simulator of Chapter 5 on x = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3] with one workgroup of 16 invocations. The table prints the 16 slots of the shared array at each barrier; slots from index s up are not touched by the step of stride s, so they keep their old values.

| moment | shared array s[0 .. 15] |
|---|---|
| after the load | [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3] |
| after the step of stride 8 | [8, 4, 9, 9, 14, 16, 11, 9, 5, 3, 5, 8, 9, 7, 9, 3] |
| after the step of stride 4 | [22, 20, 20, 18, 14, 16, 11, 9, 5, 3, 5, 8, 9, 7, 9, 3] |
| after the step of stride 2 | [42, 38, 20, 18, 14, 16, 11, 9, 5, 3, 5, 8, 9, 7, 9, 3] |
| after the step of stride 1 | [80, 38, 20, 18, 14, 16, 11, 9, 5, 3, 5, 8, 9, 7, 9, 3] |

The store phase writes part[0] = 80 (the sum of x is 80). Barriers between phases: 5. Running the invocations of every phase in descending instead of ascending order gives 80: same result, yes. With only the first 13 numbers (N = 13, the three missing slots are loaded as 0, the identity of the sum) part[0] = 61 against the direct sum 61.
<!-- /worked -->

Why is it right? After the step of stride $s$, the slot $l < s$ holds the sum of all $x_j$ with $j \equiv l \pmod s$. Before the first step ($s = n_{\mathrm{wg}}$) that is $x_l$ itself. If it holds for stride $2\,s$, the step adds slot $l + s$, which holds the sum over $j \equiv l + s \pmod{2\,s}$, to slot $l$, which holds the sum over $j \equiv l \pmod{2\,s}$; together they are all $j \equiv l \pmod s$. At the last step $s = 1$ every $j$ qualifies and slot 0 holds the total. The kernel has no race inside a step: it writes only slots below $s$, each by one invocation, and reads slots $l$ and $l + s$ where $l + s \ge s$ is never written in the step. That is why the descending order of the invocations in the trace gives the same answer as the ascending one, and why a barrier is needed *between* steps only: the reads of a step must see the writes of the one before.

The cost of the kernel, counted: $n_{\mathrm{wg}} - 1$ additions, $\log_2 n_{\mathrm{wg}}$ steps, $\log_2 n_{\mathrm{wg}} + 1$ barriers. The first step has half of the invocations adding, the second a quarter, and so on: the invocations $n_{\mathrm{wg}}/2 \ldots n_{\mathrm{wg}} - 1$ do nothing after their first step and the whole workgroup is occupied for $\log_2 n_{\mathrm{wg}}$ steps in order to do $n_{\mathrm{wg}} - 1$ additions. For $n_{\mathrm{wg}} = 256$ that is $255$ additions in $256 \times 8 = 2{,}048$ invocation-steps, 12.5 percent. A tree is fast in depth and wasteful in occupation, and the way out (give every invocation many numbers to add first, and keep the tree for the last, small stage) is the subject of the section on the second pass.

## Three ways to address the slots

The tree fixes which values are added, but not which invocation adds them. The choice interacts with the subgroups and the banks of Chapters 4 and 6, and the cost differs severalfold with the same additions. Call the kernel above kernel C. Its form is **sequential addressing**: the active invocations are the first $s$ ones, and consecutive invocations touch consecutive slots. The older textbook form (the first kernel of the progression in Harris 2007) runs the strides upward and adds neighbours first. Written naively, invocation $l$ is active when $l \bmod 2s = 0$ (kernel A):

```text
    stride = 1
    while stride < n_wg:
        if l mod (2 * stride) == 0:           # kernel A: every second, fourth, eighth invocation
            s[l] = s[l] + s[l + stride]
        barrier
        stride = stride * 2
```

The active invocations of A are spread over the whole workgroup. A subgroup of 32 consecutive invocations then has some lanes adding and some idle (divergence, Chapter 4), in every subgroup, for the first five steps. Kernel B keeps the same tree but renumbers the workers, so that the active ones are the first ones: invocation $l$ works on the slot $i = 2\,s\,l$ while $i < n_{\mathrm{wg}}$. B has no divergence between subgroups, but its slots are $2\,s$ words apart, and Chapter 6's banks answer that: a stride of two, four or eight words puts several lanes on one bank. Count both with the simulator and the models of those chapters, for a workgroup of 256 invocations (eight subgroups):

<!-- worked: reductions.addressing_steps -->
One workgroup of 256 invocations (8 subgroups of 32) reduces 256 numbers; the three kernels differ only in which invocation does the addition of a step. The simulator ran each and logged the shared-memory addresses of workgroup 0. Per step and variant the cell shows: subgroups issuing the addition (some lane active) / of them divergent (some lane idle) / worst bank-conflict degree of the step's three shared accesses (two reads and a write, the banks of Chapter 6: 32 banks of 4-byte words, word a in bank a mod 32).

| step | stride s | active invocations | share of the workgroup active | A: interleaved, l mod 2s = 0 | B: interleaved, index 2sl | C: sequential, l < s |
|---:|---|---:|---:|---|---|---|
| 1 | 1 (A, B), 128 (C) | 128 | 1/2 | 8 / 8 / 1 | 4 / 0 / 2 | 4 / 0 / 1 |
| 2 | 2 (A, B), 64 (C) | 64 | 1/4 | 8 / 8 / 1 | 2 / 0 / 4 | 2 / 0 / 1 |
| 3 | 4 (A, B), 32 (C) | 32 | 1/8 | 8 / 8 / 1 | 1 / 0 / 8 | 1 / 0 / 1 |
| 4 | 8 (A, B), 16 (C) | 16 | 1/16 | 8 / 8 / 1 | 1 / 1 / 8 | 1 / 1 / 1 |
| 5 | 16 (A, B), 8 (C) | 8 | 1/32 | 8 / 8 / 1 | 1 / 1 / 8 | 1 / 1 / 1 |
| 6 | 32 (A, B), 4 (C) | 4 | 1/64 | 4 / 4 / 1 | 1 / 1 / 4 | 1 / 1 / 1 |
| 7 | 64 (A, B), 2 (C) | 2 | 1/128 | 2 / 2 / 1 | 1 / 1 / 2 | 1 / 1 / 1 |
| 8 | 128 (A, B), 1 (C) | 1 | 1/256 | 1 / 1 / 1 | 1 / 1 / 1 | 1 / 1 / 1 |
<!-- /worked -->

<!-- worked: reductions.addressing_totals -->
Totals over the 8 steps for a workgroup of 256. Issue slots = subgroup-steps issuing the addition x 32 lanes; lane utilization = additions (255) / issue slots; shared cycles = sum over issuing subgroup-steps of the bank degrees of the two reads and the write; the last column is the same count with no conflicts. The barriers are those between the 10 phases (load, 8 steps, store).

| variant | barriers | subgroup-steps issuing | of them divergent | issue slots | lane utilization (%) | shared cycles | cycles without conflicts |
|---|---:|---:|---:|---:|---:|---:|---:|
| A: interleaved, l mod 2s = 0 | 9 | 47 | 47 | 1,504 | 17.0 | 141 | 141 |
| B: interleaved, index 2sl | 9 | 12 | 5 | 384 | 66.4 | 141 | 36 |
| C: sequential, l < s | 9 | 12 | 5 | 384 | 66.4 | 36 | 36 |
<!-- /worked -->

Read the totals. A and C do the same 255 additions with the same barriers. A issues the addition in 47 subgroup-steps, every one of them divergent, so that 17.0 percent of the issue slots do useful work; C issues it in 12, five divergent (the last five steps, when fewer than 32 invocations remain), 66.4 percent. B issues like C, but its three shared accesses per step cost 141 cycles in the bank model against C's 36: at stride $s = 4$ the 32 lanes of a subgroup read words $8l$, which fall on four banks, a conflict of degree 8, the very pattern of the register-tiled kernels of Chapter 6. Sequential addressing is conflict-free because consecutive invocations touch consecutive words, and it keeps the active invocations together, so whole subgroups are either busy or absent; the idle ones cost nothing except that they must still reach the barrier. It is the form brain's kernels use (the section on brain below).

![Two balanced binary trees over the sixteen numbers of the first example. In the top tree, kernels A and B, neighbours are added first: 3 and 1 give 4, 4 and 1 give 5, and so on up to 31, 49 and the root 80. In the bottom tree, kernel C, the leaves appear in the order 0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15, so 3 and 5 give 8, 5 and 9 give 14, up to 42, 38 and the root 80.](img/tree-reduction.svg)

The figure shows what the three kernels build. A and B are the same tree: the same pairs are added, only the invocation doing it differs. C is another tree: it adds the number $l$ to the number $l + n_{\mathrm{wg}}/2$ first, so its leaves, read left to right, are in the order of the bits of the index reversed. Both have four levels and fifteen additions, and for whole numbers they give the same 80. For float32 numbers they do not always give the same bits, which is the subject of the section on why a tree and a loop give different sums.

## More than one workgroup: the second pass

Arrays have millions of elements and a workgroup has at most a few hundred invocations. We split the array, reduce each piece in its own workgroup and combine the pieces, and we find why the combination cannot happen in the same dispatch. Give each workgroup its own slice and let it write one **partial** sum, `part[w]`. The $n_{\mathrm{grp}}$ partials must then be added too, and the natural wish is to have the last workgroup to finish do it, or all workgroups to wait at a barrier across the grid and then one of them to continue. Chapter 4 ruled both out: workgroups run in any order, on any compute unit, and nothing in a dispatch orders one against another. It is worse than "not available". A grid-wide barrier is correct only if every workgroup of the grid is resident while the others arrive, and the number of resident workgroups is fixed by the occupancy model of Chapter 4:

<!-- worked: reductions.single_dispatch_limit -->
A reduction kernel of 256 invocations, 24 registers per invocation and 1,024 bytes of shared memory per workgroup fills 8 workgroups per compute unit by the model of Chapter 4 (`gpu.occupancy`: limited by invocation slots, 2,048 // 256), so the 80 compute units hold 640 workgroups at once. A barrier across the whole grid is only safe if every workgroup of the grid is resident while it waits. With one element per invocation:

| n | workgroups | resident at once | all resident (a grid barrier could work) | waves of workgroups |
|---|---:|---:|---:|---:|
| 2^16 | 256 | 640 | yes | 1 |
| 2^18 | 1,024 | 640 | no | 2 |
| 2^20 | 4,096 | 640 | no | 7 |
| 2^22 | 16,384 | 640 | no | 26 |
| 2^24 | 65,536 | 640 | no | 103 |
<!-- /worked -->

With one element per invocation, 2^16 numbers fit the machine in one wave of 256 workgroups, and 2^18 do not: 1,024 workgroups against 640 resident. In that case the workgroups that got a slot reach the barrier and wait, none of them finishes, and so no slot is freed for the rest: a deadlock. The device may start workgroups in any order, which is exactly the freedom that lets one program run on devices of every size, and a program that depends on all of them running together depends on the size of the device. (An atomic add of each partial into one cell avoids the barrier but is the cell of the first section again: a queue, in an unspecified order. The portable reference shaders of brain use no atomics at all.)

The rule that survives is the one the host already has: the device runs the dispatches of a queue in the order they were enqueued (Chapter 5), so that what one dispatch wrote is complete when the next one reads it. The end of a dispatch is the grid-wide barrier. A **two-pass reduction** uses it: the first dispatch reduces the array to one partial per workgroup, the second dispatch reduces the partials to one number. The first pass also fixes the waste of the previous section. An invocation does not load one number; it adds many numbers in a loop, a **grid-stride loop**, in which the next number is a whole grid away, so that at every step the invocations of a workgroup read consecutive words (coalesced, Chapter 4) and the grid size can be chosen independently of the array size:

```text
kernel reduce_pass(x: array<f32>[N], part: inout array<f32>[n_grp], N: int, n_grp: int) workgroup n_wg:
    shared s: array<f32>[n_wg]
    acc = 0.0
    i = w * n_wg + l
    while i < N:
        acc = acc + x[i]
        i = i + n_grp * n_wg                # the next number, a whole grid away
    s[l] = acc
    barrier
    stride = n_wg // 2
    while stride >= 1:
        if l < stride:
            s[l] = s[l] + s[l + stride]
        barrier
        stride = stride // 2
    if l == 0:
        part[w] = s[0]

launch reduce_pass(x, part, N, n_grp) over n_grp workgroups      # pass 1
launch reduce_pass(part, total, n_grp, 1) over 1 workgroup        # pass 2
```

The second launch is the same kernel over the partials with a single workgroup, whose grid-stride loop handles any number of partials, so two dispatches suffice for every $N$. What changes with the number of elements $n_{\mathrm{el}}$ that each invocation adds is the balance between the passes, and it can be counted for $n = 2^{24}$ numbers with workgroups of 256:

<!-- worked: reductions.two_pass -->
The sum of n = 2^24 = 16,777,216 fp32 numbers in two dispatches. Pass 1: 256 invocations per workgroup, each adds n_el numbers spaced a whole grid apart (a grid-stride loop, n_el - 1 additions), then the workgroup's tree adds the 256 results (255 additions) and writes one partial. Pass 2: one workgroup of 256 invocations adds the partials, each invocation ceil(partials / 256) of them, then a tree adds the 256 results. 'Depth' counts the chain of dependent additions of both passes; 'waves' is ceil(workgroups / 640 resident); bytes = 4 x (words read + words written) with each word counted once per dispatch: pass 1 reads n and writes one partial per workgroup, pass 2 reads the partials and writes 1.

| n_el | workgroups of pass 1 | waves | additions in all | depth | bytes moved in all | bytes above the minimum 4n (%) |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 65,536 | 103 | 16,777,215 | 271 | 67,633,156 | 0.781 |
| 4 | 16,384 | 26 | 16,777,215 | 82 | 67,239,940 | 0.195 |
| 16 | 4,096 | 7 | 16,777,215 | 46 | 67,141,636 | 0.049 |
| 64 | 1,024 | 2 | 16,777,215 | 82 | 67,117,060 | 0.012 |
| 128 | 512 | 1 | 16,777,215 | 144 | 67,112,964 | 0.006 |

Every row does n - 1 = 16,777,215 additions on 4n = 67,108,864 bytes of input: 0.2500 operations per byte, against the ridge point 50 of the model GPU (Chapter 3), so a fraction 0.0050 of its peak is all a reduction can use: its cost is its bytes.
<!-- /worked -->

Every row performs the same $n - 1$ additions. The bytes are $4n$ for the array plus the partials written and read back: 0.781 percent over the minimum when each invocation adds one number (65,536 partials) and 0.006 percent when each adds 128 (512). The depth has a minimum between: with 16 numbers per invocation the chain is 46 additions, because a short loop and a short second pass add up to less than either extreme. And the number of waves of workgroups falls from 103 to one as the loop grows, which is the fewer-and-fatter choice of Chapter 4: enough workgroups to fill the machine once, and no more.

The last sentence of the table is the one to remember. A reduction performs one addition per four bytes read, a quarter of an operation per byte, the intensity of the naive matrix product of Chapter 6 and 0.005 of the peak of the model GPU, which has a ridge point of 50. However the tree is arranged, the best a reduction can do is to read its input once at the bandwidth of the device. Depth was the problem of the single cell; once the tree is in place it is the bytes that count, and the whole of the second pass and of the partials is under one percent of them. (The second dispatch has a cost the table does not show, the launch itself, which the next chapter counts.)

## Why a tree and a loop give different sums

Every tree above adds the numbers in a different order from the loop. In exact arithmetic that is invisible. In float32 it is not, and a kernel's order is part of what it computes. Volume I, Chapter 15 gave the bound for a sum of $n$ terms: each addition may round, and a term that passes through $m$ additions before reaching the total picks up $m$ rounding factors $1 + \delta$ with $|\delta| \le u = 2^{-24}$. In the left fold the first two terms pass through $n - 1$ additions, so the bound is $(n - 1)\,u \sum_i |x_i|$; in a tree every term passes through exactly $\lceil \log_2 n \rceil$ additions and the bound is $\lceil \log_2 n \rceil\,u \sum_i |x_i|$ (Higham 1993 analyses the general case). What do the errors look like in practice? Seeded numbers, six vectors of $2^{18}$ uniform numbers and the constant $0.1$:

<!-- worked: reductions.summation_error -->
Relative error |computed - exact| / exact of a float32 sum, in units of u = 2^-24, against the bounds (n - 1) u of the left fold and ceil(log2 n) u of the tree (Volume I, Chapter 15). Every addition is rounded to float32. 'Uniform' data: 6 vectors of 262,144 numbers x_i = float32(`uniform()`), drawn vector after vector in order from one generator; the first n of each vector are summed and the table prints the mean error over the 6 vectors. Generator: the book's SplitMix64, seed 0xdb25091fdd691a22 (the seed of the name `reductions.summation_error`). 'Constant' data: n copies of float32(0.1) = 0.10000000149011612. 'Left fold' adds x_0, x_1, ... in order; 'tree' adds adjacent pairs layer by layer; 'lanes' is the arrangement of a workgroup: invocation j (of 256) adds x_j, x_(j+256), ... left to right, then a sequential-addressing tree adds the 256 results. The exact sum is computed in integer arithmetic.

| n | bound (n - 1) | bound ceil(log2 n) | uniform: left fold | uniform: tree | uniform: lanes | constant: left fold | constant: tree |
|---|---:|---:|---:|---:|---:|---:|---:|
| 16 | 15 | 4 | 0.792 | 0.574 | 0.304 | 2.5 | 0 |
| 64 | 63 | 6 | 1.44 | 0.409 | 0.648 | 10 | 0 |
| 256 | 255 | 8 | 2.97 | 0.737 | 0.459 | 40 | 0 |
| 1,024 | 1,023 | 10 | 4.72 | 0.733 | 0.489 | 162 | 0 |
| 4,096 | 4,095 | 12 | 7.23 | 0.32 | 0.809 | 646 | 0 |
| 16,384 | 16,383 | 14 | 20.1 | 0.493 | 0.86 | 2.59e+03 | 0 |
| 65,536 | 65,535 | 16 | 27 | 0.305 | 0.415 | 1.04e+04 | 0 |
| 262,144 | 262,143 | 18 | 63.3 | 0.697 | 0.918 | 4.15e+04 | 0 |
<!-- /worked -->

![Error of a float32 sum against its length on log-log axes. The worst-case bound of the left fold, (n - 1) u, rises with slope 1 to 262,143; the bound of the tree rises to 18. The left fold on the constant 0.1 follows the first bound at about a sixth of its height, reaching 41,500. The left fold on uniform numbers rises more slowly, to 63. The tree and the 256-lane arrangement on uniform numbers stay below 1 for every n.](img/summation-error.svg)

On the uniform numbers the left fold's error grows from 0.79 $u$ at $n = 16$ to 63.3 $u$ at $n = 2^{18}$, a factor of 80 for a factor of 16,384 in length: about the square root, because roundings of both signs partly cancel. The tree and the workgroup arrangement stay between 0.3 and 0.9 $u$ at every length, they do not grow at all within the table. The constant series shows why the bound is not just pessimism: adding $0.1$ to a total that is no longer a multiple of the spacing rounds the same way each time, the errors add instead of cancel, and the error of the left fold reaches 41,500 $u$ at $2^{18}$, about 0.16 $n\,u$, within a factor of 6 of the bound's slope. (The tree on that constant is exactly 0: every level adds two equal numbers, which is exact.) A sum of many numbers is more accurate as a tree than as a loop, for free; this is the reason a parallel sum is usually the better one, and Chapter 15's pairwise summation is this tree.

The same fact has a consequence that the accuracy does not. Different trees give different bits. Take 200 seeded vectors of 256 numbers and add each with tree A, tree C and the left fold:

<!-- worked: reductions.tree_shapes -->
200 vectors of 256 numbers, x = float32(`uniform()`), vector by vector in order. Generator: the book's SplitMix64, seed 0x759a46498e5ea383 (the seed of the name `reductions.tree_shapes`). Each vector is summed by the left fold, by tree A (pairs of neighbours: kernels A and B of the previous blocks) and by tree C (slot l adds slot l + s, s = 128, ..., 1), all in float32. The simulator of Chapter 5, running kernels A, B and C with every addition rounded to float32, returns the same bits as the direct functions on vector 0: yes.

| pair of results | bitwise different on |
|---|---:|
| A vs C | 76 of 200 |
| A vs left fold | 160 of 200 |
| C vs left fold | 160 of 200 |

Mean relative error against the exact sum, in units of u = 2^-24:

| order | mean error / u |
|---|---:|
| left fold | 2.84 |
| A (and B) | 0.548 |
| C | 0.538 |
<!-- /worked -->

Kernels A and B build one tree and C another, and on 76 of the 200 vectors their float32 results differ; the simulator's rounded run of all three kernels returns the same bits as the direct functions, so the difference is in the order and not in an implementation detail. All three trees are better than the left fold (a mean error of about 0.54 $u$ against 2.84 $u$), and none of them is "the" correct sum. The exact sum is not even what a test should demand, as Chapter 6 argued; what matters is that a result is *reproducible*. A fixed order gives the same bits on every run, and the order of a two-pass reduction is fixed by four numbers: the size of the workgroup, the number of workgroups, the number of elements per invocation and the kernel's addressing. Change any of them and the bits may change:

<!-- worked: reductions.launch_shape_bits -->
The same n = 65,536 numbers x_i = float32(`normal()`), drawn in order from the generator of the name `reductions.launch_shape`, summed in different orders, every addition rounded to float32. Generator: the book's SplitMix64, seed 0x0117bc53c7cfbaee (the seed of the name `reductions.launch_shape`). The exact sum rounded to float32 is 92.241394 (bits 42b87b98); 'ulps' is the distance of a result from it in float32 steps. 'Two passes' is the kernel of the chapter: pass 1 has n_wg workgroups of the given size, invocation t of workgroup w adds the n_el numbers x[(k n_wg + w) size + t], k = 0 .. n_el - 1, left to right, and a sequential-addressing tree adds the workgroup; pass 2 is one workgroup of 64 invocations, invocation t adding partials t, t + 64, ... left to right (missing ones count as 0), then the same tree. The three 'one cell' rows model an unordered accumulate: the updates arrive in the order of a shuffle of 0 .. n - 1 (the book's Fisher-Yates), three shuffles drawn one after another from the generator of the name `reductions.atomic_orders` (seed 0x9c61a598f0f95f83). 'Same on rerun' repeats the whole computation.

| order | result | bits | ulps from exact | same on rerun |
|---|---:|---:|---:|---:|
| left fold | 92.2408981 | 42b87b57 | -65 | yes |
| tree of adjacent pairs | 92.2414093 | 42b87b9a | +2 | yes |
| two passes: workgroup 256, 64 workgroups, n_el = 4 | 92.2414093 | 42b87b9a | +2 | yes |
| two passes: workgroup 64, 128 workgroups, n_el = 8 | 92.2414169 | 42b87b9b | +3 | yes |
| two passes: workgroup 256, 16 workgroups, n_el = 16 | 92.2413712 | 42b87b95 | -3 | yes |
| one cell, update order = shuffle 1 | 92.2423096 | 42b87c10 | +120 | - |
| one cell, update order = shuffle 2 | 92.2409744 | 42b87b61 | -55 | - |
| one cell, update order = shuffle 3 | 92.2421722 | 42b87bfe | +102 | - |
<!-- /worked -->

Read the rows. The left fold lands 65 float32 steps from the exact sum and the trees within three. The three launch shapes of the same two-pass kernel give three different results (+2, +3 and -3 steps) for identical inputs. And the three runs of the unordered cell, which model an atomic accumulate whose arrival order the device chooses, land at +120, -55 and +102 steps: the same program on the same data, run three times, could return any of these. This is the cost the first section's cell hides beyond its depth: not only slow but irreproducible. A reduction that must be bit-reproducible fixes its order, and so fixes its shape: the number of workgroups must not depend on how many compute units the device has, which is why brain's gradient-norm kernel derives its grid from the size of the tensor alone (`gradnorm_parts`, the section on brain below). A test that compares a device sum with a reference to the last bit tests the order.

Not every reduction has the problem. The maximum, and integer sums, are exact under any regrouping. The position of the maximum is, too, provided its operation is a proper one: the pair $(v, i)$ with the rule "larger value wins, and on a tie the lower index wins" is associative and commutative, so every tree agrees with the left scan. The rule has to be written into the combination when the invocations own *strided* indices, because then the left operand of a combination is no longer the one with the lower index:

<!-- worked: reductions.argmax_ties -->
x = [2, 9, 3, 1, 9, 4, 0, 5, 6, 2, 1, 7] (N = 12); the maximum 9 occurs at indices 1 and 4. A pair is (value, index); the sequential scan keeps the first maximum. Partial results of the 4 invocations, strided ownership: (9, 4), (9, 1), (3, 2), (7, 11); run ownership: (9, 1), (9, 4), (6, 8), (7, 11). Trees use sequential addressing (invocation l combines its pair with that of l + s, s = 2, ..., 1, the left operand being invocation l).

| method | (value, index) returned |
|---|---|
| left to right scan, lower index wins ties | (9, 1) |
| 4 invocations own indices t, t + 4, ...; tree; 'keep the left on a tie' | (9, 4) |
| 4 invocations own indices t, t + 4, ...; tree; 'lower index wins a tie' | (9, 1) |
| 4 invocations own runs of 3 (indices 3 t .. 3 t + 2); tree; 'keep the left on a tie' | (9, 1) |
<!-- /worked -->

The simple rule "keep the left operand unless the right is strictly larger", correct for a left scan, returns index 4 when invocation 0 owns indices 0, 4, 8 and invocation 1 owns 1, 5, 9: the combination sees the pair from invocation 0 on the left, with the higher index. Written with the index in the comparison it returns the scan's index 1. When each invocation owns a contiguous run, the left operand always has the lower indices and the simple rule is enough, which is the arrangement of the two-stage argmax of brain.

## Parallel scans

Volume I, Chapter 16 used the prefix sum as the offset of each item's output, and said where the sort and the packing would need it. A kernel needs it computed by many invocations at once, a **parallel scan**. We build two and count which one to use when. The doubling scan of that chapter runs in rounds: in round $k$ every position adds the value $2^k$ places to its left, and after $\log_2 n$ rounds the inclusive scan is done, at a cost of 17 additions for eight numbers where the sequential scan uses seven. Written as a kernel it is the **Hillis-Steele scan** (Hillis and Steele 1986). One new difficulty appears: in a round every position reads a value that another position writes in the same round, so a position first reads into a register, then all wait, then all write:

```text
kernel scan_hillis_steele(x: array<u32>[n_wg], y: inout array<u32>[n_wg]) workgroup n_wg:
    shared a: array<u32>[n_wg]
    a[l] = x[l]
    barrier
    s = 1
    while s < n_wg:
        v = a[l]
        if l >= s:
            v = v + a[l - s]
        barrier                          # every read of this round is done
        a[l] = v
        barrier                          # every write is done before the next round reads
        s = s * 2
    y[l] = a[l]
```

<!-- worked: reductions.scan_trace_hillis -->
Input x = [3, 1, 4, 1, 5, 9, 2, 6] (Volume I, Chapter 16), one workgroup of 8 invocations, run by the simulator of Chapter 5. Sequential inclusive scan [3, 4, 8, 9, 14, 23, 25, 31]. The shared array at each step (every step reads, waits at a barrier, writes, waits again):

| moment | shared array |
|---|---|
| input | [3, 1, 4, 1, 5, 9, 2, 6] |
| stride 1 | [3, 4, 5, 5, 6, 14, 11, 8] |
| stride 2 | [3, 4, 8, 9, 11, 19, 17, 22] |
| stride 4 | [3, 4, 8, 9, 14, 23, 25, 31] |

Output [3, 4, 8, 9, 14, 23, 25, 31]; equals the sequential inclusive scan: yes. Additions 17, barriers 7.
<!-- /worked -->

The rounds are the rows of Chapter 16's table, 17 additions in all: $n \log_2 n - n + 1$ for a power of two, since round $s$ adds at $n - s$ positions. Its depth is $\log_2 n$ and it is the shortest possible. Its work is $\Theta(n \log n)$, though, against $n - 1$ for the sequential scan. When there are more numbers than workers the extra work is paid for in steps, and a scan that does not waste work is worth having.

The **Blelloch scan** (Blelloch 1990) does $2(n - 1)$ additions. It runs the reduction tree first, in place (the up-sweep), so that the slots at the positions $2^k m - 1$ hold the sums of the aligned blocks they end. It then clears the root and walks the tree down (the down-sweep): each node hands its left child its own value, the prefix before it, and gives its right child that prefix plus the left child's sum.

```text
kernel scan_blelloch(x: array<u32>[n_wg], y: inout array<u32>[n_wg]) workgroup n_wg:
    shared a: array<u32>[n_wg]
    a[l] = x[l]
    barrier
    s = 1
    while s < n_wg:                          # up-sweep: the reduction tree, in place
        if l < n_wg // (2 * s):
            i = 2 * s * (l + 1) - 1
            a[i] = a[i] + a[i - s]
        barrier
        s = s * 2
    if l == 0:
        a[n_wg - 1] = 0                      # the prefix before the first element is empty
    barrier
    s = n_wg // 2
    while s >= 1:                            # down-sweep: pass prefixes down the tree
        if l < n_wg // (2 * s):
            i = 2 * s * (l + 1) - 1
            t = a[i - s]
            a[i - s] = a[i]                  # the left child gets the prefix of its parent
            a[i] = a[i] + t                  # the right child gets that prefix plus the left sum
        barrier
        s = s // 2
    y[l] = a[l]
```

<!-- worked: reductions.scan_trace_blelloch -->
The same input x = [3, 1, 4, 1, 5, 9, 2, 6], one workgroup of 8 invocations (at most 4 of them add in any step). Sequential exclusive scan [0, 3, 4, 8, 9, 14, 23, 25]. The shared array at each barrier:

| moment | shared array |
|---|---|
| input | [3, 1, 4, 1, 5, 9, 2, 6] |
| up-sweep s = 1 | [3, 4, 4, 5, 5, 14, 2, 8] |
| up-sweep s = 2 | [3, 4, 4, 9, 5, 14, 2, 22] |
| up-sweep s = 4 | [3, 4, 4, 9, 5, 14, 2, 31] |
| root cleared | [3, 4, 4, 9, 5, 14, 2, 0] |
| down-sweep s = 4 | [3, 4, 4, 0, 5, 14, 2, 9] |
| down-sweep s = 2 | [3, 0, 4, 4, 5, 9, 2, 23] |
| down-sweep s = 1 | [0, 3, 4, 8, 9, 14, 23, 25] |

Output [0, 3, 4, 8, 9, 14, 23, 25]; equals the sequential exclusive scan: yes. Additions 14, barriers 8.
<!-- /worked -->

![The Blelloch scan of 3, 1, 4, 1, 5, 9, 2, 6 as eight rows of eight boxes. Rows two to four are the up-sweep: 4, 5, 14, 8, then 9 and 22, then 31, each from two arrows. The fifth row is the root set to 0. Rows six to eight are the down-sweep, with solid arrows for sums and dashed arrows for the copy of a parent's prefix to its left child, ending in 0, 3, 4, 8, 9, 14, 23, 25.](img/blelloch-scan.svg)

The result is the *exclusive* scan, which is the one that gives offsets. The up-sweep is Chapter 5's reduction in a different layout (7 additions) and the down-sweep is the same tree walked back (another 7), with the depth doubled to $2 \log_2 n$. Counted by the simulator, over seeded integers of four sizes, and by the formulas for larger ones:

<!-- worked: reductions.scan_costs -->
Both scans in one workgroup of n invocations on seeded integers, checked against the sequential scan (inclusive for Hillis-Steele, exclusive for Blelloch). Generator: the book's SplitMix64, seed 0x355f69e43fa0bfa4 (the seed of the name `reductions.scan_check`). Each size draws its n values with `integer(100)` in order, the sizes one after another. 'Additions' are counted by the simulator; the formula columns are n log2 n - n + 1 and 2 (n - 1).

| n | Hillis-Steele equal | additions | formula | barriers | Blelloch equal | additions | formula | barriers |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | yes | 17 | 17 | 7 | yes | 14 | 14 | 8 |
| 16 | yes | 49 | 49 | 9 | yes | 30 | 30 | 10 |
| 64 | yes | 321 | 321 | 13 | yes | 126 | 126 | 14 |
| 256 | yes | 1793 | 1793 | 17 | yes | 510 | 510 | 18 |

Work, depth and the least number of steps on n_wk = min(n, 10,240) workers, max(ceil(N_work / n_wk), N_dep), by the formulas:

| n | algorithm | work N_work | depth N_dep | steps on n_wk workers (at least) |
|---|---|---:|---:|---:|
| 1,024 | sequential | 1,023 | 1,023 | 1,023 |
| 1,024 | Hillis-Steele | 9,217 | 10 | 10 |
| 1,024 | Blelloch | 2,046 | 20 | 20 |
| 1,048,576 | sequential | 1,048,575 | 1,048,575 | 1,048,575 |
| 1,048,576 | Hillis-Steele | 19,922,945 | 20 | 1,946 |
| 1,048,576 | Blelloch | 2,097,150 | 40 | 205 |
<!-- /worked -->

Neither is better everywhere. At $n = 1{,}024$ on 1,024 workers Hillis-Steele needs 10 steps and Blelloch 20: depth rules, and the doubled depth loses. At $n = 2^{20}$ on the 10,240 lanes of the model GPU the work rules: $19.9$ million additions are at least 1,946 steps, while Blelloch's $2.1$ million are at least 205, nine and a half times fewer, with its depth of 40 irrelevant. Inside one workgroup of at most 256 invocations either works and Hillis-Steele has fewer barriers per addition of work; across a long array, the work-efficient one wins. (The Blelloch scan's slots are $2\,s$ words apart, so it has the bank pattern of kernel B; padding the shared array, as Chapter 6 did for tiles, removes it, and Harris, Sengupta and Owens 2007 describe the layout.)

A scan longer than a workgroup meets the wall of the previous section again and has the same cure, applied recursively. It is the blocked scan of Chapter 16: scan every block independently and record its total; scan the totals; add each block's scanned total back to its block. Each phase is a dispatch, because each needs the one before it to be finished. The totals are a shorter array, so the second phase is itself a scan, until they fit one block:

<!-- worked: reductions.block_scan_dispatches -->
The exclusive scan of n u32 words in blocks of 256: a dispatch with one invocation per block scans its block serially in place and writes the block's total; the totals are scanned the same way, recursively, until they fit one block; then one dispatch per level above the first adds the scanned totals back. 'Levels' lists the number of words scanned at each level. Dispatches = levels block scans + (levels - 1) add-backs. 'Chain' = the longest chain of dependent additions (each block scan adds up to 255 times in sequence, each add-back once). Words are counted once per dispatch (block scan: read n_j, write n_j + totals; add-back: read n_j + totals, write n_j); the minimum for a scan is 2n (read every word, write every word).

| n | levels (words scanned) | dispatches | invocations in the first dispatch | chain | words moved | ratio to the minimum 2n |
|---|---|---|---:|---:|---:|---:|
| 2^16 | 65,536 -> 256 | 3 | 256 | 511 | 263,169 | 2.008 |
| 2^20 | 1,048,576 -> 4,096 -> 16 | 5 | 4,096 | 527 | 4,218,945 | 2.012 |
| 2^24 | 16,777,216 -> 65,536 -> 256 | 5 | 65,536 | 767 | 67,503,105 | 2.012 |
<!-- /worked -->

For $2^{20}$ numbers in blocks of 256 there are three levels (1,048,576, 4,096 and 16 words scanned), five dispatches in all, and about $4n$ words moved: twice the minimum $2n$ that a scan must at least read and write, since each element is written by the block scan and read and written again by the add-back. A scan that reached the minimum in one pass would have each block wait for its predecessor's total (the decoupled look-back of Merrill and Garland 2016), which needs the forward-progress guarantee between workgroups that the core rule does not give. The table's first dispatch has 4,096 invocations for $2^{20}$ numbers, fewer than the 10,240 lanes: here the block of 256 serial additions per invocation costs parallelism, a choice that the section on brain shows in the code.

A scan of floating-point numbers is reordered too, with the consequences of the previous section; the scans that kernels need most are scans of counts and offsets, which are integers, exact, and the same whatever the order.

## Sorting by digits

Volume I, Chapter 16 sorted integer keys without comparing them, by a histogram, a scan and a scatter, one digit at a time. Each of the three parts must now run on many invocations without any two writing the same word.

The sequential **radix sort** of Chapter 16 (Merrill and Grimshaw 2011 describe a tuned GPU form) made a pass per digit, from the least significant: count the keys with each digit value (the histogram), take the exclusive scan of the counts (the first position of each digit's block), and place each key at its position and increment it; the placement is stable, so the order of an earlier digit survives. In parallel the count and the placement are the problems, since all keys increment the same counters. The answer is to give each invocation a **chunk** of $L_{\mathrm{ch}}$ consecutive keys and its own private row of counters, so that no counter is shared and no atomic is needed. With $n_{\mathrm{ch}} = \lceil n / L_{\mathrm{ch}} \rceil$ chunks and $2^{b_{\mathrm{dg}}}$ digit values per pass, the histogram $\mathrm{hist}[d \cdot n_{\mathrm{ch}} + j]$ is the number of keys of chunk $j$ with digit $d$, stored *digit-major*: all chunks of digit 0, then all chunks of digit 1, and so on. One exclusive scan of that whole array then gives, in position $(d, j)$, the number of keys with a smaller digit, plus the number of keys with digit $d$ in the chunks before $j$: exactly where chunk $j$ has to start writing its keys with digit $d$. The scatter walks each chunk in order from its starting offsets, so keys with equal digits keep their order inside a chunk (the walk is in order) and across chunks (the offsets are in chunk order): stable.

```text
kernel radix_hist(keys: array<u32>[n], hist: inout array<u32>[R * n_ch], shift: int, n: int, n_ch: int, L_ch: int, R: int)
        workgroup n_wg:
    if g >= n_ch: return
    count = zeros(u32, [R])
    for i in g * L_ch .. min((g + 1) * L_ch, n):
        d = (keys[i] >> shift) and (R - 1)
        count[d] = count[d] + 1                          # a private row: no other invocation writes it
    for d in 0 .. R:
        hist[d * n_ch + g] = count[d]

kernel radix_scatter(keys_in: array<u32>[n], vals_in: array<u32>[n], hist: array<u32>[R * n_ch],
        keys_out: inout array<u32>[n], vals_out: inout array<u32>[n], shift: int, n: int, n_ch: int, L_ch: int, R: int)
        workgroup n_wg:
    if g >= n_ch: return
    off = zeros(u32, [R])
    for d in 0 .. R:
        off[d] = hist[d * n_ch + g]                      # where this chunk starts writing digit d
    for i in g * L_ch .. min((g + 1) * L_ch, n):
        d = (keys_in[i] >> shift) and (R - 1)
        keys_out[off[d]] = keys_in[i]
        vals_out[off[d]] = vals_in[i]
        off[d] = off[d] + 1
```

Each pass is three dispatches: `radix_hist`, a scan of the histogram (the multi-block scan of the previous section), and `radix_scatter`, with the key buffers exchanged between passes. The simulator of Chapter 5 runs it on 16 seeded keys of 6 bits with 3-bit digits and chunks of 4:

<!-- worked: reductions.radix_trace -->
16 keys of 6 bits (0 .. 63), each paired with its input position; digits of 3 bits (8 values), 2 passes from the least significant digit; chunks of 4 keys, so 4 chunks and one invocation per chunk. Generator: the book's SplitMix64, seed 0xb4d4f68165e69cd0 (the seed of the name `reductions.radix_trace`). The keys are 16 calls of `integer(64)` in order: [21, 1, 32, 24, 41, 43, 9, 0, 2, 48, 55, 61, 12, 5, 9, 22]. Pass 0 histogram: entry (d, j) is the number of keys of chunk j with digit d, stored at hist[4 d + j] (digit-major); the scan of the whole array is the position where chunk j writes its first key with digit d.

| digit d | count, chunk 0 | count, chunk 1 | count, chunk 2 | count, chunk 3 | start, chunk 0 | start, chunk 1 | start, chunk 2 | start, chunk 3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2 | 1 | 1 | 0 | 0 | 2 | 3 | 4 |
| 1 | 1 | 2 | 0 | 1 | 4 | 5 | 7 | 7 |
| 2 | 0 | 0 | 1 | 0 | 8 | 8 | 8 | 9 |
| 3 | 0 | 1 | 0 | 0 | 9 | 9 | 10 | 10 |
| 4 | 0 | 0 | 0 | 1 | 10 | 10 | 10 | 10 |
| 5 | 1 | 0 | 1 | 1 | 11 | 12 | 12 | 13 |
| 6 | 0 | 0 | 0 | 1 | 14 | 14 | 14 | 14 |
| 7 | 0 | 0 | 1 | 0 | 15 | 15 | 15 | 16 |

The pairs key:position after each pass:

| moment | pairs |
|---|---|
| input | 21:0 1:1 32:2 24:3 41:4 43:5 9:6 0:7 2:8 48:9 55:10 61:11 12:12 5:13 9:14 22:15 |
| after pass 0 (digit 0, bits 0 .. 2) | 32:2 24:3 0:7 48:9 1:1 41:4 9:6 9:14 2:8 43:5 12:12 21:0 61:11 5:13 22:15 55:10 |
| after pass 1 (digit 1, bits 3 .. 5) | 0:7 1:1 2:8 5:13 9:6 9:14 12:12 21:0 22:15 24:3 32:2 41:4 43:5 48:9 55:10 61:11 |

The keys of the last pass are sorted: yes; equal keys keep their input order (positions equal the stable order): yes. Words moved by the simulated dispatches: 224 loads + 194 stores = 418; the formula 5n + 2 N_hist + S(N_hist) per pass with n = 16, N_hist = 32 histogram words and S(N_hist) = 65 words of the scan gives 418: yes.
<!-- /worked -->

The histogram table is the whole mechanism in one view. Digit 0 has four keys, held two by chunk 0, one by chunk 1 and one by chunk 2, and the scan places chunk 0's first key with digit 0 at position 0, chunk 1's at 2, chunk 2's at 3 and chunk 3's (it has none) at 4, which is where digit 1's block begins at chunk 0 (4). After pass 0 the pairs are ordered by the low digit with the original positions in ascending order within each digit (the two keys equal to 9 stay in the order 6, 14); after pass 1 by the whole key. The last lines of the block count the words the three dispatches moved and compare them with a formula, which can then be used for sizes that cannot be traced by hand.

Counting a pass shows what the histogram costs. With keys and values, one pass reads the keys once to count, writes the histogram ($N_{\mathrm{hist}} = 2^{b_{\mathrm{dg}}} n_{\mathrm{ch}}$ words), scans it, and then reads keys, values and histogram and writes keys and values: $5n + 2 N_{\mathrm{hist}} + S(N_{\mathrm{hist}})$ words, where $S$ counts the scan (about $4 N_{\mathrm{hist}}$ for a long histogram, by the previous table). The histogram is the new cost, and it grows with the number of chunks. Two choices, the digit width and the chunk length, trade the passes against it:

<!-- worked: reductions.radix_costs -->
The words a radix sort of 32-bit keys with 32-bit payloads moves, counted by the rule of the trace above: one pass moves 5n + 2 N_hist + S(N_hist) words, N_hist = 2^b_dg n / L_ch histogram words for digits of b_dg bits and chunks of L_ch keys, S(N_hist) the words of the 256-element block scan of N_hist words (previous block).

First, n = 2^20 keys and L_ch = 1,024 keys per chunk (1,024 chunks), the digit width varying; passes = ceil(32 / b_dg):

| b_dg | passes | N_hist | N_hist / n | words per pass | words in all | in all / n |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 32 | 2,048 | 1/512 | 5,255,201 | 168,166,432 | 160.4 |
| 2 | 16 | 4,096 | 1/256 | 5,267,521 | 84,280,336 | 80.4 |
| 4 | 8 | 16,384 | 1/64 | 5,341,441 | 42,731,528 | 40.8 |
| 6 | 6 | 65,536 | 1/16 | 5,637,121 | 33,822,726 | 32.3 |
| 8 | 4 | 262,144 | 1/4 | 6,821,905 | 27,287,620 | 26.0 |
| 11 | 3 | 2,097,152 | 2 | 17,875,073 | 53,625,219 | 51.1 |
| 16 | 2 | 67,108,864 | 64 | 409,475,089 | 818,950,178 | 781.0 |

Then n = 2^24 keys, b_dg = 8 (four passes), the chunk length L_ch varying; 'lane fill' is min(1, chunks / 10,240) with 10,240 lanes:

| L_ch | chunks n_ch (invocations) | N_hist / n | words in all / n | lane fill |
|---|---:|---:|---:|---:|
| 256 | 65,536 | 1 | 44.09 | 1.00 |
| 512 | 32,768 | 1/2 | 32.05 | 1.00 |
| 1,024 | 16,384 | 1/4 | 26.02 | 1.00 |
| 2,048 | 8,192 | 1/8 | 23.01 | 0.80 |
| 4,096 | 4,096 | 1/16 | 21.51 | 0.40 |
| 8,192 | 2,048 | 1/32 | 20.75 | 0.20 |
<!-- /worked -->

For $2^{20}$ keys and 1,024 chunks, digits of 8 bits move about $26 n$ words in four passes, the least of the widths tried: narrower digits mean more passes (40.8 $n$ at 4 bits), wider ones a histogram that outgrows the keys (51.1 $n$ at 11 bits, with a histogram twice as long as the input). The chunk length is a second compromise. A short chunk gives many invocations and a histogram as large as the input; at $n = 2^{24}$ and 256 keys per chunk the histogram has $n$ words and a sort moves $44.1 n$, against $23.0 n$ at 2,048 keys. A longer chunk lowers the words and, past 2,048, begins to leave lanes idle (a fill of 0.80, then 0.40). The words keep falling as the chunk grows (20.8 $n$ at 8,192 keys), so they alone would say "longer"; what stops the growth is the lanes, and where it stops depends on the device.

Sorting is sometimes more than the job needs. A sampler wants the $k$ largest of a vocabulary of $V$ logits, not their order, and sorting everything costs the four passes above. Taking the maximum $k$ times, masking the winner each time, costs $k$ reductions of $V$ words:

<!-- worked: reductions.topk_vs_sort -->
Selecting the k largest of V = 2^17 = 131,072 logits. Sort: the radix sort of the previous block, 32-bit keys, four passes of b_dg = 8, chunks of L_ch = 256 keys (so N_hist = 131,072 histogram words), 5,779,492 words in all. Rounds: k two-pass argmax reductions, each reading V words, writing 256 (value, index) pairs, reading them and writing the winner (132,097 words a round; masking the winner is one word and not counted). The cheaper way is the one with fewer words; the rounds stop being cheaper from k = 44.

| k | rounds: words | rounds / V | sort / V | fewer words |
|---:|---:|---:|---:|---|
| 1 | 132,097 | 1.0 | 44.1 | rounds |
| 2 | 264,194 | 2.0 | 44.1 | rounds |
| 4 | 528,388 | 4.0 | 44.1 | rounds |
| 8 | 1,056,776 | 8.1 | 44.1 | rounds |
| 16 | 2,113,552 | 16.1 | 44.1 | rounds |
| 32 | 4,227,104 | 32.3 | 44.1 | rounds |
| 64 | 8,454,208 | 64.5 | 44.1 | sort |
<!-- /worked -->

For $V = 2^{17}$ the rounds move fewer words than the sort for every $k$ up to 43 and more from 44: a $k$ of 1 to 16, the usual range of sampling and of a mixture-of-experts router, costs 2 to 37 percent of the sort's words. The comparison counts words only, and rounds have a launch cost each (the next chapter's subject); the choice between them is a choice of what to count, as Chapter 16 said.

## Reductions, scans and sorts in brain's kernels

All of the chapter's kernels exist in `crates/kernels/wgsl/`, in the shapes the repository's constraints allow: core compute, no atomics, no subgroups (`AGENTS.md`, quoted in Chapter 4), and a barrier only where the device reports `DeviceCaps::workgroup_reductions`.

**The tree in a workgroup.** `moe_router_topk.wgsl` is kernel C: a workgroup of 256 invocations per row of router logits, thread `t` owning the experts `t`, `t + 256`, ..., reducing the maximum and the softmax denominator with `for s = 128; s > 0; s >>= 1` and `if t < s`, sequential addressing in the code. Its top-$k$ selection is $k$ rounds of the same tree on (value, index) pairs, and the comparison writes the tie rule of this chapter explicitly (`ov == red_v[t] && oi < red_i[t]`), as it must when the invocations own strided experts. The header counts the saving as the chapter does: a row costs $k \cdot \log_2 256$ barrier steps against $k^2 n_{\mathrm{experts}}$ serial steps of the kernel it replaces, and it notes that the softmax denominator is now summed in a tree, so a weight can differ from the older kernel's by an ulp. `moe_route_count.wgsl` counts one expert's slots in the same tree, and `moe_route_emit.wgsl` places them with an inclusive Hillis-Steele scan of 256 run counts, read into a register, barrier, write, barrier, the pseudocode of this chapter. The kernels in between choose differently: `moe_route_scan.wgsl` scans the expert counts on *one* thread, because the header says there are a few hundred experts and one thread is cheaper than a parallel scan's extra launches, the work-and-depth count with a launch added. The test `the_routing_tables_are_a_stable_counting_sort_of_the_ids` in `crates/gpu-core/tests/it/moe_i8_grouped.rs` checks the three kernels against a host counting sort (commit 4084aa37d added them; it states that they are stable, atomic-free and need no host readback).

**One barrier, then a fold.** Most row kernels do not run the halving loop. `rmsnorm_rows.wgsl` accumulates a strided partial per invocation into a 64-slot shared array, hits one barrier, and has every invocation fold the 64 partials in a serial loop; `softmax_rows.wgsl` does the same for its maximum and its sum, and `gradnorm_part.wgsl` has invocation 0 do the fold. The headers of the first and the last give the reason: a fold of 64 additions is cheaper than a second barrier, and the CPU JIT of the repository supports one top-level barrier. In the language of the chapter it trades depth (63 additions in a chain instead of 6) for barriers (1 instead of 7), and the order of the additions is fixed, which keeps the result reproducible.

**Two passes: the argmax and the norm.** `argmax_part.wgsl` and `argmax_final.wgsl` are a two-pass reduction on (value, index) pairs: each invocation scans one contiguous chunk of a row (the run ownership above, so the plain rule suffices, and the header says that both stages break ties to the lowest index), and `crates/model/src/readout.rs` fixes 256 chunks per row (`ARGMAX_CHUNKS`). The same file builds the top-$k$ list the way the last table counts it: `cap` rounds of argmax followed by masking the winner (`topk_extract_step.wgsl`), tested in `crates/model/tests/it/readout_topk.rs`. The pair came with commit e60ec8151 (July 2026), which replaced a single-thread scan of each row of logits; its message reports that scan as a tenth of a decode step in a profile.

<!-- perf-number: historical measurement, commit e60ec8151 -->
In July 2026, on a Tesla P40, in a batched decoding benchmark of a small synthetic model, the commit's profile put the single-thread argmax over 32k logits at 10.3 percent of decode.

**A one-lane sum: commit dba351816.** The gradient-norm of the optimizer is a sum of squares over every parameter. `gradnorm_sq.wgsl` is the cell of this chapter's first section made into a program: `if (gidx != 0u) { return; }`, one invocation walking the whole tensor, depth equal to work. Commit dba351816 (July 2026) replaced it by `gradnorm_part.wgsl` and `clip_coef_wg.wgsl`: the first gives each tensor $n_{\mathrm{grp}} = \min(\lceil \mathrm{numel} / 8192 \rceil, 512)$ workgroups of 64 (`gradnorm_parts` in `crates/paramstore/src/lib.rs`) that add with a grid-stride loop and write one partial each, the second folds all partials in one workgroup: a two-pass reduction exactly as above. For the largest tensor of the commit's table, $38{,}597{,}376$ numbers, the chain on one lane is that many dependent additions and the loop of one invocation of the new kernel is $\lceil 38{,}597{,}376 / (512 \cdot 64) \rceil = 1{,}178$ additions. The commit measured the step:

<!-- perf-number: historical measurement, commit dba351816 -->
In July 2026, on a Tesla P40, training a six-layer GPT of width 768 for five steps, the commit reports the one-lane kernel at 30,133 ms of 34,545 ms of device time, 87.2 percent, and at 14.2 ms after the change.

Its message also makes a claim about accuracy, that against a float64 oracle the serial walk was off by 2.34e-3 (relative) at 4.19 million elements and the tree by 2.40e-7. This chapter cannot confirm those two numbers, since the data of the commit is not ours, but it shows how a left fold can be that far off: 2.34e-3 is 39,000 $u$, the size that the constant series of the error table reaches (41,500 $u$ at $2^{18}$) and far above what random numbers do (63 $u$). The second half of the story is knowledge #132: a kernel selection that opts in *by name* kept running `gradnorm_sq` on a model whose kernel list never named the cooperative pair, with identical numbers and a training step an order of magnitude slower, until a test (`crates/decide/tests/it/optimizer_kernels.rs`) asserted that the pair is registered. The selection in `crates/backend-api/src/select.rs` (`Op::GradNorm`) puts `SplitReduction` first only where `workgroup_reductions` holds; knowledge #26 is the other side of the same gate, a barrier kernel recorded on the CPU backend, which corrupted memory instead of refusing.

**The scan and the sort: commits bd714622f, 242b55c98 and 8eb2d2196.** `scan_block.wgsl` and `scan_add.wgsl` are the blocked scan with one invocation per block of 256 words, scanning its block serially, and the recursion is in `crates/gpu-core/src/sort.rs` (`record_scan`: the tower of block-sum buffers, one level per factor of 256, as the dispatch table counts). `sort_hist.wgsl` and `sort_scatter.wgsl` are the digit-major histogram and the stable scatter of this chapter with 8-bit digits, and `record_sort_pairs` loops them over the passes. Commit bd714622f (July 2026) introduced all four as brain's first scan and sort, for tile-based rasterization, under the rules of no atomics, no subgroups and a fixed workgroup of 64; its message says that these multi-pass dispatches replace what a CUDA library does with atomics. The code's own comment on the chunk length records the second commit: a sort of millions of keys with 256-key chunks launched tens of thousands of invocations each holding a 256-bin table, and the histogram scan grew with them; commit 242b55c98 (November 2026) made the chunk grow with the key count, up to 2,048 (`sort_chunk_len` in `crates/gpu-core/src/sort.rs`).

<!-- perf-number: historical measurement, commit 242b55c98 -->
In November 2026, on frames of 10 to 25 million keys (the hardware is not in the commit's message; a code comment on `SORT_CHUNK_MAX` names a GH200), the commit reports the sort and scan kernels at 12.3 s and then 7.1 s of a 44.8 s device profile.

The radix table above explains the direction: at 256 keys per chunk the histogram is as large as the keys (44.1 $n$ words), at 2,048 an eighth of that (23.0 $n$), a ratio of 1.9 in words against the commit's 1.7 in time on its hardware, one model against one measurement. Commit 8eb2d2196 (November 2026) is the third part: the scan and sort had lived in the splat crate, so the grouped mixture-of-experts forward kept a private copy of the scan; it moved both to `gpu_core::sort` for every crate, and the correctness gate for both is `crates/gpu-core/tests/it/scan_sort.rs`, which runs them against a plain iterator scan and a stable sort, and checks that a scan shorter than its scratch reports the right total.

The lesson common to these stories is the first section's: a single invocation is a queue. In each case the repair was to find the chain that was as long as the work, cut it, and keep the order of the additions a function of the shapes alone.

## Check your understanding

1. A reduction of one million numbers on 10,240 lanes has work 999,999 and depth 20. Which of $\lceil N_{\mathrm{work}} / n_{\mathrm{wk}} \rceil$ and $N_{\mathrm{dep}}$ is the binding lower bound, and what does that say about making the tree shallower?
2. A colleague says kernels A and C "do the same additions, so they cost the same". Using the totals table, say what is right and what is wrong in the sentence.
3. Find the error: a tree kernel writes `if l >= stride: return` at the top of each step so that "the idle half does not reach the barrier".
4. Explain why a single dispatch cannot add $2^{24}$ numbers with a grid-wide barrier on the model GPU, and say what changes if the device had room for all 65,536 workgroups at once.
5. A test compares the sum of a kernel on a laptop GPU with the sum on a workstation GPU, bit for bit, and fails. The kernel launches as many workgroups as there are compute units. Explain, and give two ways to make the results equal.
6. The argmax tree above, with strided ownership and the rule "keep the left unless the right is strictly larger", returns index 4 for the vector of the example. Where does it go wrong, and why does the same rule work for the two-stage argmax with contiguous chunks?
7. For $n = 2^{20}$ on the model GPU, which scan, Hillis-Steele or Blelloch, needs fewer steps at least, by the formulas, and why does the answer reverse for $n = 1{,}024$?
8. Find the error: "in the radix sort the histogram can be stored chunk-major, hist[j * R + d], and a single scan over it still gives every chunk's starting offsets."

## Summary

A sum of $n$ numbers computed by every invocation updating one cell has depth $n$: the cell serves one update at a time, in an order nobody controls. A tree has the same $n - 1$ additions and depth $\lceil \log_2 n \rceil$, and with $n_{\mathrm{wk}}$ workers cannot take fewer than $\max(\lceil N_{\mathrm{work}} / n_{\mathrm{wk}} \rceil, N_{\mathrm{dep}})$ steps, which for a million numbers on 10,240 lanes is 103 and for the cell a million. In a workgroup the tree is a loop with a barrier per step, correct by a short invariant, and the choice of which invocation does each addition decides the cost: sequential addressing issues the addition in 12 subgroup-steps and 36 shared cycles for 256 invocations, where the textbook interleaved form issues it in 47 and, renumbered, costs 141 cycles of bank conflicts. Past one workgroup a dispatch cannot synchronize its workgroups (a grid barrier deadlocks once the grid outgrows the resident workgroups), so the array is reduced in two passes, with a grid-stride loop that keeps every invocation busy and moves under one percent more bytes than the input, at a quarter of an operation per byte, bandwidth-bound like the naive matrix product.

The tree changes the order of the additions. Its error bound is $\lceil \log_2 n \rceil u$ against $(n - 1) u$, and the seeded example shows the left fold at 63 $u$ and the trees below 1 $u$ for $2^{18}$ numbers. Different trees give different bits, a reduction's order is fixed by its shape, and an unordered accumulate returns different results on identical runs; a maximum or an argmax with a proper tie rule is exact under any order. The prefix sum has two parallel forms, Hillis-Steele with $n \log n$ work and depth $\log n$, and Blelloch with $2(n-1)$ work and depth $2 \log n$, the first better for short arrays and the second, 205 steps against 1,946 for a million numbers, for long ones; a scan longer than a workgroup takes one dispatch per level and about twice the minimum words. The radix sort is private chunk histograms, one scan of the digit-major array and a stable scatter, with a word count that picks 8-bit digits and chunks long enough that the histogram stays small, and a selection of the $k$ largest by $k$ rounds of argmax costs fewer words than the sort up to $k = 43$ for 131,072 logits.

What these programs have in common is that they are short, bandwidth-bound kernels, each followed by another. A softmax, a normalization or a gradient norm is such a reduction between two elementwise kernels, and every kernel boundary writes its result to memory and reads it back. The next chapter counts what that costs, and what fusing the kernels saves.

## References

- Richard P. Brent, "The parallel evaluation of general arithmetic expressions", 1974.
- W. Daniel Hillis and Guy L. Steele Jr., "Data parallel algorithms", 1986.
- Guy E. Blelloch, "Prefix sums and their applications", 1990.
- Nicholas J. Higham, "The accuracy of floating point summation", 1993.
- Mark Harris, "Optimizing parallel reduction in CUDA", 2007.
- Mark Harris, Shubhabrata Sengupta and John D. Owens, "Parallel prefix sum (scan) with CUDA", 2007.
- Duane Merrill and Andrew Grimshaw, "High performance and scalable radix sorting", 2011.
- Duane Merrill and Michael Garland, "Single-pass parallel prefix scan with decoupled look-back", 2016.

## Exercises

1. **[Calculate]** For $n = 10^7$ numbers, workgroups of 256 and $n_{\mathrm{el}} = 64$, find the number of workgroups of pass 1, the partials, the words read and written by each pass, the additions in all and the depth of the chain, with the counting rules of `reductions.two_pass`.
2. **[Calculate]** For a workgroup of 64 invocations (two subgroups), find for kernels A, B and C the number of subgroup-steps that issue the addition and how many of them are divergent.
3. **[Calculate]** Find the bank degree of the first read of kernel B at each step for a workgroup of 128 invocations, and the shared cycles of the whole reduction (two reads and a write per issuing subgroup-step).
4. **[Calculate]** Find the work, the depth and the least number of steps on 4,096 workers for the sequential, Hillis-Steele and Blelloch scans of $n = 2^{16}$, and the smallest $n$ at which Blelloch's bound is the smaller on 4,096 workers.
5. **[Calculate]** For $n = 10^7$ words and blocks of 1,024, find the levels of the multi-block scan, the dispatches, the words moved and the ratio to $2n$.
6. **[Calculate]** For $n = 2^{22}$ keys, 16-bit keys with 8-bit digits and chunks of 1,024 keys, find the words of the radix sort by the formula of the chapter, and the number of chunks.
7. **[Implement]** Reproduce `reductions.tree_kernel_trace` and `reductions.addressing_totals` with a simulator of your own (phases separated by barriers; a log of the shared addresses of workgroup 0), then add the totals for a workgroup of 64 and compare with exercise 2.
8. **[Implement]** Reproduce one row of `reductions.summation_error` (the left fold, the adjacent tree and the 256-lane arrangement for $n = 4096$ on the first of the six vectors) with the generator and draw order printed with it, using correctly rounded float32 addition and an exact integer sum for the reference.
9. **[Implement]** Reproduce `reductions.launch_shape_bits` for the left fold, the adjacent tree and the first two-pass shape, and add a shape of your own (a workgroup of 128, 32 workgroups, 16 numbers per invocation); report which bits change.
10. **[Implement]** Reproduce `reductions.scan_trace_hillis`, `reductions.scan_trace_blelloch` and the first four rows of `reductions.scan_costs` with one implementation of each scan and a counter of the additions.
11. **[Implement]** Reproduce `reductions.radix_trace` (the three dispatches per pass, the digit-major histogram) and then sort 16 keys of 12 bits with 4-bit digits, comparing with your language's stable sort.
12. **[Investigate]** Run the sum of $2^{22}$ float32 numbers in a thread pool of your own with 1, 2, 4 and 8 threads, each thread adding a contiguous slice and the slices added in thread order. Report which thread counts give identical bits, and what a fixed number of slices independent of the thread count does to the answer.

## Key terms

- **parallel reduction**: a reduction algorithm carried out by many invocations at once, whose cost is described by its work (the operations it performs) and its depth (the longest chain of operations that must wait for one another). <!-- concept: parallel-reduction -->
- **tree reduction**: a parallel reduction that combines the values in pairs, then the pairs in pairs, so that $n$ values need $n - 1$ combinations in only $\lceil \log_2 n \rceil$ dependent steps; inside a workgroup it is a loop over strides with a barrier after each step. <!-- concept: tree-reduction -->
- **two-pass reduction**: the reduction of an array longer than a workgroup by two dispatches, one in which every workgroup writes a partial result for its slice and a second that reduces the partials, the end of the first dispatch serving as the barrier between workgroups that a dispatch cannot have. <!-- concept: two-pass-reduction -->
- **parallel scan**: a prefix sum computed by many invocations in a number of steps that grows with $\log n$, either with $n \log n$ additions (Hillis-Steele) or with $2(n - 1)$ additions and twice the depth (Blelloch), and across workgroups by scanning blocks, scanning their totals and adding the totals back. <!-- concept: parallel-scan -->
- **radix sort**: a sort of integer keys digit by digit from the least significant, each pass made of per-chunk histograms, one scan of the digit-major histogram array and a stable scatter, so that no two invocations write the same counter or the same position. <!-- concept: radix-sort -->

## Answers

**Recall questions.**

1. $\lceil \log_2 1000 \rceil = 10$ layers, $999$ additions (Volume I, Chapter 16).
2. $[0, 2, 9, 10]$. The doubling scan uses $8 \cdot 3 - 8 + 1 = 17$ additions for $n = 8$ (Volume I, Chapter 16).
3. $1$ and $1 + 2^{-23}$. The first sum $1 + 2^{-24}$ is a tie between 1 and $1 + 2^{-23}$ and rounds to the even 1, twice; the second adds the small numbers first to the representable $2^{-23}$ (Volume I, Chapter 15).
4. No. Workgroups run in any order on any compute unit, and nothing in one dispatch orders one against another; the second may not even have started (Volume IV, Chapter 4).
5. $\lfloor 65536 / (24 \cdot 64) \rfloor = 42$ by registers, $\lfloor 65536 / 512 \rfloor = 128$ by shared memory, $2048 / 64 = 32$ by slots, a cap of 16: 16 workgroups, 1,024 invocations, occupancy 50 percent (Volume IV, Chapter 4).
6. Only if every invocation of the workgroup takes the return, so that none reaches the barrier. If some return and others reach it, the others wait for invocations that never arrive and the result is undefined; the guard belongs around the memory accesses (Volume IV, Chapter 5).

**Check your understanding.**

1. $\lceil 999{,}999 / 10{,}240 \rceil = 98$ and the depth is 20, so the work per worker binds; the steps are between 98 and 118. A shallower tree would not change the bound: the way to fewer steps is less work per worker (more workers), and the first lever on a real device is the bytes, not the depth.
2. Right: the same 255 additions and the same 9 barriers. Wrong: A issues the addition in 47 subgroup-steps, all divergent (17.0 percent of issue slots useful), C in 12 (66.4 percent); additions are not the cost, subgroup-steps are.
3. The invocations that return do not reach the next barrier, and the others wait for them: the barrier rule of Chapter 5. The `if l < stride` must surround the addition only, and every invocation must reach the barrier at the end of the step.
4. $2^{24}$ numbers with one per invocation need 65,536 workgroups against 640 resident. The resident ones reach the barrier, wait, and never finish, so the other 64,896 never get a slot: a deadlock. If the device held all 65,536 at once the barrier would work on that device, and the program would then be correct only there; the two-pass form works everywhere.
5. The number of workgroups changes with the device, so the grid-stride loop and the partials are different, and the order of the additions, so the bits. Fix the grid (a number of workgroups that depends only on $n$, as brain does for the gradient norm) or compare to a tolerance of the order of $\lceil \log_2 n \rceil u \sum |x_i|$ instead of exactly.
6. Invocation 0 owns indices 0, 4, 8, so its best pair is $(9, 4)$, and invocation 1 owns 1, 5, 9 and has $(9, 1)$. The final combination has invocation 0's pair on the left and keeps it on the tie, returning the higher index 4. With contiguous chunks the left operand always has lower indices than the right, so keeping the left on a tie is the scan's rule.
7. Hillis-Steele needs at least 1,946 steps and Blelloch 205 for $n = 2^{20}$, because the work divides by 10,240. At $n = 1{,}024$ on 1,024 workers the work does not bind: Hillis-Steele needs 10 (its depth) and Blelloch 20.
8. A scan of the chunk-major array puts all digits of chunk 0 before those of chunk 1, so chunk 0 would write all its keys, of every digit, before chunk 1's: the output is grouped by chunk and not by digit. The digit-major layout makes the order of the scan digit first, then chunk.
