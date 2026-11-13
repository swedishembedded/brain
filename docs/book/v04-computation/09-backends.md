<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# One Program, Many Devices

*How can one set of kernels run on CPUs, Vulkan GPUs and CUDA GPUs with the same results?*

Take the kernel of Chapter 5, $y_g = 2 x_g + 1$ over ten numbers, written once as a WGSL text. Three machines are to run it: a GPU that is driven through Vulkan, a GPU that is driven through CUDA, and the cores of a CPU. They do not accept the same program. The Vulkan driver takes a binary intermediate form of the kernel and compiles it for its own chip. The CUDA toolchain takes C++ text with its own names for the indices (`blockIdx`, `threadIdx`) and compiles that. A CPU has no launch at all: it needs loops. So the one text has to be turned into three different programs, and the three programs must put the same ten numbers in $y$.

Here is the source, the CUDA text a translator produces from it, the loop nest that a CPU compiler produces from it, a line-by-line map between them and the result of running each form on the book's models of the three devices. The uniform block is the list of 32-bit words that travels to every device ($n$, then the bit patterns of the two floats), and the invocations of the first model run in a seeded shuffled order because the kernel may not depend on the order:

<!-- worked: backends.three_paths -->
The kernel y = scale x + shift of Chapter 5, scale = 2, shift = 1, N = 10, x = [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6], workgroup size 4. The uniform block is the list of 32-bit words [10, 0x40000000, 0x3f800000] (n, then the bit patterns of scale and shift). Generator: the book's SplitMix64, seed 0x327b4c00edc16855 (`seed_of` of the name `backends.three_paths`, defined on the pseudocode page). Draw: one Fisher-Yates shuffle (the book's) of the 12 invocation numbers 0 .. 11, giving the order [5, 10, 1, 0, 4, 6, 3, 7, 9, 2, 8, 11] (invocation number = w * 4 + l).

The WGSL source:

```wgsl
struct Params {
    n: u32,
    scale: f32,
    shift: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let g = gid.x;
    if (g >= p.n) { return; }
    y[g] = p.scale * x[g] + p.shift;
}
```

The same kernel as the translator gives it to a CUDA device (names and the clamp are the emitter's; the temporaries it introduces for sub-expressions are left out):

```cuda
extern "C" __global__ void brain_scale_shift(const unsigned int* __params,
    float* __b1, float* __b2, unsigned long long __n1, unsigned long long __n2)
{
  const size_t __nel1 = __n1 / 1u;
  const size_t __nel2 = __n2 / 1u;
  const unsigned int __gid_x = blockIdx.x * 4u + threadIdx.x;
  if (__gid_x >= __brain_uu32(__params, 0u)) {
    return;
  }
  __b2[__brain_clamp(__gid_x, __nel2)] = __brain_uf32(__params, 4u) * __b1[__brain_clamp(__gid_x, __nel1)] + __brain_uf32(__params, 8u);
}
```

The same kernel as a CPU loop nest, in the pseudocode of the book:

```text
function scale_shift_cpu(x: array<f32>[N], y: inout array<f32>[N], scale: f32, shift: f32, N: int, n_grp: int):
    for w in 0 .. n_grp:                  # workgroups: the loop that is shared out over the cores
        for l in 0 .. 4:                      # the invocations of one workgroup, in order
            g = w * 4 + l
            if g >= N: continue           # `return` ends this invocation only
            y[g] = scale * x[g] + shift
```

| construct | WGSL | translated for CUDA | CPU loop nest |
|---|---|---|---|
| workgroup size | `@workgroup_size(4)` | `blockDim` = 4 at launch, `blockIdx.x * 4u` | inner loop bound 4 |
| global index g | `gid.x` | `blockIdx.x * 4u + threadIdx.x` | `g = w * 4 + l` |
| uniform p.n | `p.n` (u32 at byte 0) | `__brain_uu32(__params, 0u)` | the argument `N` |
| uniform p.scale | `p.scale` (f32 at byte 4) | `__brain_uf32(__params, 4u)`, bit pattern reread as f32 | the argument `scale` |
| x[g] and y[g] | `x[g]`, `y[g]` | `__b1[__brain_clamp(g, __nel1)]` | `x[g]`, `y[g]` |
| `return` | `return;` | `return;` | `continue` |

| execution | program form | workgroups or blocks | times the body ran | y equals scale x + shift |
|---|---|---:|---:|---|
| invocation model, shuffled order | text compiled by the driver | 3 | 12 | yes |
| CUDA-like model, last block first | translated C++ text | 3 | 12 | yes |
| CPU loop nest | loops, no launch | 3 | 12 | yes |

y = [-5, -3, -1, 1, 3, 5, 7, 9, 11, 13].
<!-- /worked -->

Read the map before the listings. The line `if (g >= p.n) { return; }` stays a guard in the CUDA text and becomes a `continue` in the loop nest, because a `return` ends one invocation and an iteration of a loop is what an invocation has become. The index $g = w \cdot 4 + l$ is `gid.x` on the first device, `blockIdx.x * 4u + threadIdx.x` on the second and the loop counters on the third. Two things that the source never says appear in the CUDA text. Every access to `x` and `y` goes through `__brain_clamp`, an index limited to the length of its buffer, and the uniform block is read by byte offset, `__brain_uf32(__params, 4u)`, the offset the WGSL layout rules give the field `scale`. Neither is decoration. Chapter 5 showed what the guard is for and what a host and a kernel that disagree about byte offsets exchange; the question now is what a *translator* must do so that those promises of the language are still true in the target.

Three programs from one text raise four questions. What does the code above the devices see, so that it never has to know which one it runs on (the backend abstraction)? What must a translation preserve, and where do translations break (shader translation)? When is the translation compiled to machine code, at the moment the program starts or before it ships (just-in-time or ahead-of-time compilation)? And what does it mean that the three devices give "the same results", given that floating-point arithmetic is not associative and a fused multiply-add rounds once where two operations round twice (cross-backend parity)?

The comparisons ahead lean on six earlier ideas, some of which the first example already uses.

1. (Chapter 5, the guard.) A kernel is launched over $N = 10$ elements with workgroups of 4 invocations. How many invocations run, and what does the guard `if g >= N: return` do for them?
2. (Chapter 5, the barrier rule.) When is a `return` before a `barrier` legal, and what happens to the others when it is not?
3. (Chapter 2, the CPU core.) A float32 sum is taken once with one accumulator and once with eight lanes. Why can the two results differ, and which has the smaller error bound?
4. (Chapter 8, fused multiply-add.) What does a compiler change when it contracts $x y + z$ into one instruction, and how many of 4,096 seeded triples differed?
5. (Chapter 8, command recording.) What does `submit` hand to the device, and why is a list that is still open not part of what `read_back` waits for?
6. (Volume I, Chapter 15, rounding.) A float32 sum of $n$ terms by a left fold has an error of at most about $(n - 1)\,u \sum_i |x_i|$. What is $u$ for float32, and what is the corresponding bound for a tree of adjacent pairs?

## One interface, one kernel set

The code that builds a model must not change when the device does, so the first thing to settle is what a device has to offer and what a program may ask it about itself. A **backend** is a program that provides the host operations of Chapter 5 for one kind of device. The code above it, the model, uses nothing else. The set of operations and the meaning the model may rely on is the **backend abstraction**. In the pseudocode of the book (`docs/book/pseudocode.md`) the contract is this list of signatures:

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

(A backend that records also provides `begin_commands`, `record` and `submit` of Chapter 8.) Six of the seven are the host operations the reader already used to run $y = 2x + 1$; the seventh is new. `capabilities()` is the device describing itself: the largest workgroup size a kernel may declare, the shared memory of a workgroup, the width of a subgroup (0 when the device does not say) and the largest number of barriers a kernel may contain (0, 1 or unbounded). Four rules make the interface worth having.

**Handles are opaque.** A `buffer` and a `kernel` are tokens that only the backend that made them can interpret; the model passes them back and never looks inside. On one device a buffer is memory on a card, on another a block of host memory, and the model's code is the same.

**The queue is in order and the read waits.** Every backend runs what was dispatched in the order it was dispatched, each launch seeing the writes of the ones before it, and `read_back` is the point where the host waits. These are the meanings of Chapter 5, now a duty of each backend; a backend that reorders two dependent dispatches is wrong however fast it is.

**A capability says what the device can do, and an unknown one is absent.** A kernel that uses a barrier must not be chosen for a device whose compiler allows at most zero barriers, and the choice is made by reading the record, not by asking what kind of device it is. The record reports what it cannot establish as the value that forbids the feature (0, `false`), so a missing answer can only make the engine more cautious. Chapter 10 builds the kernel selector on top of this.

**A kernel that cannot be compiled is an error.** `compile` returns a kernel that computes what the source computes or it reports an error that names the kernel. A backend that quietly runs a simpler kernel instead gives an answer that looks like the real one and is not.

The reason for the narrowness is the number of things that must be kept equal. Suppose a program has $K$ kernels and runs on $D$ devices. If each device has its own hand-written version, there are $K D$ files that all have to compute the same function, and a change to the function is a change to $D$ files. If there is one source and a translator for each device, there are $K + D$ artifacts, and a defect in the way a construct is translated is repaired once, in a translator, for all $K$ kernels:

<!-- worked: backends.maintained_artifacts -->
D = 3 devices (a CPU, a Vulkan GPU, a CUDA GPU) and K kernels. 'Per-device copies': one hand-written source per kernel per device, K D files. 'One source and translators': K sources and D translators, K + D artifacts. A comparison of two devices on one kernel is one cell of the parity matrix, K D (D - 1) / 2 cells in both designs: translation does not remove the testing, it removes the copies that can drift and moves the repair of a defect from K D files to D translators.

| K | files, per-device copies | artifacts, one source and translators | parity cells, either design |
|---|---:|---:|---:|
| 10 | 30 | 13 | 30 |
| 100 | 300 | 103 | 300 |
| 1,000 | 3,000 | 1,003 | 3,000 |
<!-- /worked -->

The number of comparisons does not shrink: two devices must still agree on every kernel, $K D (D - 1) / 2$ cells in both designs, and the last section is about those cells. What shrinks is the number of places where a person can make them disagree. The cost is a constraint on the kernels: they are written for the language that every translator understands (the subset of WGSL of Chapter 5), and a hand-tuned kernel for one device is an addition, never a replacement. Chapter 8 met the rule in one form: a fused kernel written natively for one device has no twin in the portable catalogue; it is selected by name, it returns nothing where the device or the shape is not served, and the caller then builds the chain of portable kernels that it replaces, so that the portable chain remains the reference on every device.

## Shader translation: what must survive

A translator is a compiler from one kernel language to another, and a compiler is correct when the program it produces does what the source says. What a kernel language says is mostly not in its syntax, so we begin by listing it.

**Shader translation** is rewriting a kernel from the language it is written in into the form a device accepts, in such a way that the translated program computes what the source program computes for every input on which the source says what that is. The last clause is the hard one. A kernel language is a contract with its programmers, and the contract contains promises that are never written into a kernel because the kernel relies on them. A translator is correct only if it keeps the promises that a target does not make by itself:

| what the source language and its reference promise | what a target does if left alone | what the translator must do |
|---|---|---|
| an index past the end of an array stays inside the array (clamped, zero, or a dropped store) | reads and writes whatever lies next in memory | clamp every index to the length of its buffer |
| `i32` addition, subtraction and multiplication wrap modulo $2^{32}$ | C++ calls signed overflow undefined | do the arithmetic on the unsigned bit pattern |
| a shift count outside $0 .. 31$ is indeterminate | one machine masks it, another clamps it | pick the reference's answer and write it out |
| each arithmetic operation rounds to nearest, ties to even (the reference's arithmetic) | a compiler may fuse a multiply and an add, or use a reciprocal square root | forbid contraction, and use the exact operation |
| a barrier is reached by every invocation of the workgroup | an invocation that returned never arrives | turn a return into a flag, or run the kernel as phases |
| a `return` before a barrier ends the invocation for good | a phase-by-phase execution may let it run on | carry the fact across the barrier |
| workgroup memory starts at zero | shared memory holds what the last workgroup left | zero it when the workgroup starts |
| a uniform block has WGSL's layout | C++ lays a struct out its own way | read each field at the byte offset the layout gives |
| two buffers may be the same memory | `restrict` promises they are not | never promise that |

Three of the rows can be computed and are worth seeing in numbers, because each is a place where two targets disagree without either being at fault:

<!-- worked: backends.preserved_semantics -->
**Shift counts.** The unsigned 32-bit word 1 shifted left by s. A machine that masks the count takes s mod 32; one that clamps gives 0 for s >= 32. WGSL calls the range s >= 32 indeterminate, so a translation must pick the reference's rule and write it out (here: mask).

| s | 1 << (s mod 32), masked | clamped at the word width | agree |
|---:|---:|---:|---|
| 0 | 0x00000001 | 0x00000001 | yes |
| 1 | 0x00000002 | 0x00000002 | yes |
| 31 | 0x80000000 | 0x80000000 | yes |
| 32 | 0x00000001 | 0x00000000 | no |
| 33 | 0x00000002 | 0x00000000 | no |
| 63 | 0x80000000 | 0x00000000 | no |

**Signed overflow.** WGSL's i32 arithmetic wraps modulo 2^32. The table is the wrapped value next to the exact one (C++ calls the overflow undefined, which is why a translator does the arithmetic on the unsigned bit pattern and reads the result back as signed).

| expression | exact integer | i32, wrapped | same |
|---|---:|---:|---|
| 2147483647 + 1 | 2,147,483,648 | -2,147,483,648 | no |
| -2147483648 + -1 | -2,147,483,649 | 2,147,483,647 | no |
| 65536 * 65536 | 4,294,967,296 | 0 | no |
| 100000 * 100000 | 10,000,000,000 | 1,410,065,408 | no |

**Bounds.** The kernel y[g] = x[g + 1] for g = 0 .. 5, x = [10, 20, 30, 40, 50, 60], whose last invocation reads one element past the buffer (the pattern of a ragged tile that masks the result afterwards). y[5] under the three rules a language may choose; the word after the buffer is the first word of whatever buffer the allocator placed next.

| rule for an index past the end | word after the buffer | y[5] |
|---|---:|---:|
| unchecked | 500 | 500 |
| unchecked | -777 | -777 |
| clamp to the last element | - | 60 |
| return zero | - | 0 |
<!-- /worked -->

The first table is the case where the source leaves the answer open. WGSL calls a shift count of 32 or more indeterminate, so there is no correct answer to inherit, and the two machines give different ones. The translator may not leave the choice to the target. It writes `(b & 31u)` into the CUDA text, which is what the reference implementation (the CPU compiler of the next section) computes, so that two tiers never disagree on an input the language leaves open (knowledge #114). The second table is the integer case, where C++ leaves the answer open and WGSL does not: the translator reaches the defined wrap by going through the unsigned type. The third is the bounds row, the one that the first listing above showed as `__brain_clamp`. A kernel with a ragged tile reads a few elements past the end and masks the result, relying on the clamp; a target without it returns the neighbour buffer's word, which differs from one run to the next because it depends on what the allocator put there. The failure appears only when other threads are allocating, so a test that runs alone passes (knowledge #202 records this arriving as a flaky test, and `compute-sanitizer` naming an out-of-range read as its cause).

The rows about the barrier, the return and zeroed workgroup memory are the subject of the next section, where a CPU compiler has to rebuild all three by hand. The row about rounding returns in the last section.

Notice what the table says about where bugs live. The promises are the ones the *kernels* never test: a kernel does not check that shared memory started at zero, because the language said it does. A translator that gets one wrong produces a plausible number, not an error, and a happy-path comparison on a full-size tile passes. Knowledge #112 states the discipline: when you retarget a language, list the guarantees it makes that the target does not, and write one test per guarantee.

## The CPU: a workgroup becomes loops

A CPU has no thousands of lanes and no launch, so the translator's job there is to derive a loop nest from the kernel, add the barrier, and find the three things the loops must do by hand.

The listing of the opening already holds the whole idea for a kernel without a barrier: the workgroups are an outer loop, the invocations of a workgroup an inner loop, and the body is the body. Since workgroups may run in any order, the outer loop is the one a CPU shares out among its cores, each core taking a range of workgroups, and nothing synchronizes them.

A barrier changes this. Chapter 5 defined it as a rendezvous: every invocation of the workgroup has finished the statements before it before any invocation starts the statements after it. A CPU core runs one invocation at a time, so the rendezvous is built by splitting the body at the barrier into two phases and giving each phase its own loop over the invocations. Loop 1 runs the code before the barrier for $l = 0, 1, 2, 3$; when it ends, every invocation has finished it, which is the barrier; loop 2 then runs the code after. This is exactly what the simulator of Chapter 5 did when it ran "a phase for every invocation of a workgroup before it started the next"; the simulator was the transformation, and a CPU compiler emits it as machine code instead of running it as a program. The figure shows one workgroup of the kernel below, as the GPU runs it and as the CPU compiles it:

![A workgroup of four invocations with a barrier, as a GPU and as a CPU runs it. On the GPU the four invocations run the first phase together, a barrier waits for all four, and the four run the second phase together. On the CPU the first phase is a loop over l from 0 to 3 whose end is the barrier, followed by a second loop. Both show the shared array 5, 6, 7, 8 and the sum 26 stored by invocation 0.](img/barrier-phases.svg)

The kernel is the smallest one that has everything a barrier brings with it: a shared array, a guard that returns before the barrier, and slots that some invocations never write. It sums the numbers of a workgroup in shared memory. In the pseudocode, with the semantics of WGSL (which zeroes workgroup memory at the start; the kernels of Chapters 5 to 8 never read a slot that nobody wrote, but this one does, as the real reductions of an engine do at the ragged end of their data):

```text
kernel wg_sum(x: array<f32>[N], out: inout array<f32>[n_real], N: int, n_real: int) workgroup 4:
    shared s: array<f32>[4]               # WGSL: all 0.0 when the workgroup starts
    if w >= n_real: return                # a padded grid; every invocation of the workgroup agrees
    if g < N: s[l] = x[g]
    barrier
    if l == 0: out[w] = s[0] + s[1] + s[2] + s[3]
```

The CPU compilation of it, with the three mechanisms that the loops need marked:

```text
function wg_sum_cpu(x: array<f32>[N], out: inout array<f32>[n_real], N: int, n_real: int, n_grp: int):
    s = zeros(f32, [4])                         # one array, allocated once and reused by every workgroup
    for w in 0 .. n_grp:
        for l in 0 .. 4: s[l] = 0.0             # (1) zero the shared array at the start of EACH workgroup
        alive = zeros(bool, [4])                # (2) which invocations have not returned
        for l in 0 .. 4: alive[l] = true
        for l in 0 .. 4:                        # loop 1: the code before the barrier
            if w >= n_real:
                alive[l] = false                # `return` is recorded, not forgotten
                continue
            g = w * 4 + l
            if g < N: s[l] = x[g]
        # the end of loop 1 is the barrier
        for l in 0 .. 4:                        # loop 2: the code after the barrier
            if not alive[l]: continue           # (3) a returned invocation stays returned
            if l == 0: out[w] = s[0] + s[1] + s[2] + s[3]
```

Each marked line answers one promise of the table above. (1) Without it the second workgroup reads what the first left in slots that no invocation of the second wrote. (2) and (3) carry a `return` across the split: the `continue` of loop 1 only skips the rest of loop 1, and unless loop 2 knows, the invocations that a padded-grid guard excludes run the second half and write their output. The grid here is padded, five workgroups for $N = 10$ of which three are real; $x = [1, \ldots, 10]$, and an output slot nobody writes holds $-1$. Each mistake is run in turn, the three mechanisms present and each missing, on the book's models:

<!-- worked: backends.cpu_phases -->
The kernel `wg_sum` with workgroups of 4: each invocation with g < N stores x[g] in its slot of a shared array of 4; after the barrier invocation 0 stores the sum of the 4 slots in out[w]. The grid is padded: 5 workgroups are launched for N = 10 (3 are real), and `if w >= 3: return` (a test every invocation of a workgroup passes or fails together) stands before the barrier. x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]; an out slot nobody writes holds -1. WGSL zeroes workgroup memory at the start of each workgroup, so the last real workgroup, which fills 2 of its 4 slots, sums two zeros.

| execution | out[0 .. 4] | equals GPU semantics |
|---|---|---:|
| GPU semantics: barrier is a rendezvous, shared memory zeroed per workgroup, a return is permanent | [10, 26, 19, -1, -1] | yes |
| CPU loops, return kept in a mask across both loops, shared memory cleared per workgroup | [10, 26, 19, -1, -1] | yes |
| CPU loops, a return only skips the rest of the first loop | [10, 26, 19, 0, 0] | no |
| CPU loops, shared memory reused without clearing | [10, 26, 34, -1, -1] | no |
| CPU loops, both mistakes | [10, 26, 34, 34, 34] | no |

A local variable that lives across the barrier. Kernel: a = x[g]; s[l] = 2 a; barrier; out[g] = s[(l + 1) mod 4] + a, over x = [1, 2, 3, 4, 5, 6, 7, 8] in 2 workgroups of 4.

| execution | out[0 .. 7] | equals GPU semantics |
|---|---|---:|
| GPU semantics | [5, 8, 11, 6, 17, 20, 23, 18] | yes |
| CPU loops, the local re-created in loop 2 | [4, 6, 8, 2, 12, 14, 16, 10] | no |
| CPU loops, the local promoted to a per-workgroup array | [5, 8, 11, 6, 17, 20, 23, 18] | yes |
<!-- /worked -->

The first two rows agree because they are the same program: the rendezvous model and its loop nest. With the return scoped to the first loop the padded workgroups 3 and 4 store zeros where the kernel must leave the slot alone. With the shared array not cleared, workgroup 2 (which fills two of its four slots) sums the 7 and 8 that workgroup 1 left in slots 2 and 3, and stores 34 where 19 is right. Each is a plausible number, and each appears only at a ragged end or on a padded grid. The second table shows the fourth difficulty. A value that one invocation computes before the barrier and uses after it (`a` here) cannot live in a variable that the loops re-create on every iteration: loop 2 sees a fresh `a` of 0 and the answer is wrong in every slot. The cure is to promote the variable to a per-workgroup array indexed by $l$, one slot for each invocation, which is what the last row does. A compiler may instead refuse such a kernel, and brain's does (the section on brain below).

The loops also fix a limit, which the interface reports through `capabilities().max_barriers`. A body can be cut into phases at barriers that stand at its top level, and one barrier is the case above. A barrier inside a loop, or a second barrier under some structure, would need the loop of the kernel to be turned inside out so that the inner loop over $l$ runs for every iteration of the kernel's own loop. A compiler that does not do that reports that it cannot express the kernel; the kernel then has to run through some other path on that device.

## Reaching a GPU: text, IR and a driver

On a GPU the last steps of the translation are done by someone else's compiler, so the useful question is how much of the path the engine still controls. We follow the source along it.

On a Vulkan-style device the first stages of the path are shared. A **front end** parses the WGSL, checks it (types, address spaces, rules of control flow) and produces one internal form, an intermediate representation (IR). A **back end** writes that form in the language the platform wants: SPIR-V words for Vulkan, other languages for other platforms. The GPU driver then compiles those words for the chip, at the moment the program asks for a pipeline. On a CUDA device the same internal form is written as C++ text by the translator of the previous sections, and NVRTC, a compiler that ships with the CUDA toolkit, turns the text into machine code (a *cubin*) for the architecture of the device, again at run time. A CUDA device reports its *compute capability*, a pair of integers written major.minor (6.1, 8.9, 10.0); the machine-code architecture is named `sm_` followed by the pair without the dot (`sm_61` for 6.1, `sm_100` for 10.0). The toolchain can also emit *PTX* (Parallel Thread Execution), a portable instruction text of CUDA that is not yet machine code: it is written for a virtual architecture (`compute_61`) and the driver finishes the compilation for the actual chip when the program loads. On a CPU the internal form goes to a code generator and then to the machine code of the cores, in memory, at start. The figure shows the three paths with their stages:

![The paths of one WGSL kernel to three devices. A shared front end parses and validates it into one checked IR. For a Vulkan GPU the IR is written as SPIR-V and compiled by the driver; for a CUDA GPU it is written as CUDA C++ with clamped indices and compiled by NVRTC, or loaded as a cubin built ahead of time; for a CPU it goes to a JIT compiler that produces machine code for loops. Under each device the counts for y = 2x + 1: 3 workgroups of 4 invocations, 3 blocks of 4 threads, a 3 by 4 loop nest, 12 runs of the body each.](img/one-source-three-devices.svg)

<!-- perf-number: historical measurement, knowledge #51 -->
Two facts about this path matter for everything that follows. First, every stage is a program with options, and the options are part of what is compiled. The same WGSL can be compiled with or without runtime bounds checks, with or without a guard counter on every loop; a backend that sets these differently from another computes with a different set of promises and runs a different amount of work (knowledge #51 records a factor of about two in the measured arithmetic rate of one device between two backends compiling the same text, traced to those options). A cache of compiled code that ignores the options is therefore wrong, which the next section turns into a rule. Second, the driver's compiler is not the engine's: it may, unless told otherwise, fuse a multiply and an add, and the engine can ask a CUDA compiler not to (a flag) but has no such control over every Vulkan driver. That is why the last section does not assume bit-identity across devices.

## Compiling at start and compiling before

Translating and compiling cost work, and the work is paid somewhere: the two designs below differ in where, and a cache decides how often.

**JIT compilation** (just-in-time) compiles a kernel from its source at the moment a program needs it, on the machine that will run it. Its strength is that it compiles for whatever is actually present: the architecture the device reports, the compiler version installed, the flags in force. Its price is that the work of the compiler is paid on that machine, at start, and that the machine must carry the compiler. **Ahead-of-time compilation** (AOT) compiles for a list of known architectures before the program is shipped and stores the results with a key that identifies exactly what they were built from. At start a program only loads: it needs the driver and no compiler. Its price is size (one image per kernel per architecture) and coverage (a device that is not in the list has no image).

Both designs compile with a given amount of work, which a model can count. The parameters below are the model's and not measurements: a program registers $N_{\mathrm{ker}} = 100$ kernels and one run dispatches $N_{\mathrm{used}} = 12$ of them; a kernel's source is $s_{\mathrm{src}} = 2{,}400$ bytes, one compiled image $s_{\mathrm{img}} = 6{,}000$ bytes, and a compiler spends $c_{\mathrm{cmp}} = 500$ steps per source byte. Each design is charged the compiler steps it runs and the bytes it reads (a kernel's source is read to form its key; a hit also reads the image) and writes (a miss stores the image):

<!-- worked: backends.startup_work -->
Parameters of the model (not measurements): N_ker = 100 kernels registered, N_used = 12 of them dispatched by a run, source s_src = 2,400 bytes, one compiled image s_img = 6,000 bytes, a portable (PTX-like) form 3,000 bytes, a compiler spends c_cmp = 500 steps per source byte. Work of one start, in compiler steps (c_cmp s_src for each kernel compiled), bytes read (each source is read to form its key; a hit also reads its image) and bytes written (a miss stores its image).

| design of the start | kernels handled | compiler steps | bytes read | bytes written | toolchain needed at the target |
|---|---:|---:|---:|---:|---|
| compile every registered kernel at construction, empty cache | 100 | 120,000,000 | 240,000 | 600,000 | a compiler |
| compile every registered kernel at construction, warm cache | 100 | 0 | 840,000 | 0 | a compiler on some earlier run |
| compile a kernel at its first dispatch, empty cache | 12 | 14,400,000 | 28,800 | 72,000 | a compiler |
| compile a kernel at its first dispatch, warm cache | 12 | 0 | 100,800 | 0 | a compiler on some earlier run |
| load a prebuilt image for the device (ahead of time) | 12 | 0 | 100,800 | 0 | none |

Disk for prebuilt images: N_ker (N_tgt s_img + s_ptx) bytes for N_tgt targets plus one portable form: 3,900,000 bytes for the six targets of the older toolkit lane (39,000 bytes per kernel) and 2,100,000 for the three of the newer one (21,000 per kernel).
<!-- /worked -->

Four things stand out. Compiling everything at construction costs $c_{\mathrm{cmp}}\,s_{\mathrm{src}}\,N_{\mathrm{ker}} = 120{,}000{,}000$ steps with an empty cache; compiling a kernel at its first dispatch costs only the 12 that the run uses, $14{,}400{,}000$, 12 percent of it. Which is right depends on whether a program knows what it needs: an engine that registers a whole catalogue and uses a slice of it wants the lazy form. A warm cache removes the compiler steps entirely, at the price of reading the image and the source for every kernel (840,000 bytes at construction, 100,800 lazily). The AOT start has the same bytes as the warm lazy start and the same zero steps. What it adds is the last column: it is the only design that runs where no compiler is present. And all of these still translate the WGSL to the target text at run time, because the key is formed from the text; the translation is the cheap part, the machine-code compilation the expensive one.

**The compile cache.** A **compile cache** stores the machine code a compilation produced under a key, so a later run asks for the key and skips the compiler. It is correct only if the key contains everything that can change the code. A key that misses one input serves code compiled for a different question, and for a kernel that is a wrong number and not a crash. Each input below is in the key; the table changes one at a time and shows the key change:

<!-- worked: backends.cache_key -->
The key of one compiled kernel is the SHA-256 digest (first 12 hexadecimal digits shown) of its inputs, each written as an 8-byte little-endian length and then its UTF-8 bytes, in the order: source text (491 bytes, the CUDA text of `backends.three_paths`), entry name, device architecture, compiler version, flags, specialization macros, launch ABI. Base inputs: entry `brain_scale_shift`, `sm_61`, `nvrtc 12.9`, `--fmad=false`, no macros (the field is present and empty: it is hashed as the length 0), `abi1`. Each row replaces one field: the comment row appends the line `// shift 1.5` and a newline to the source text; the entry row is `brain_other`; the macro row is `-DTILE=16`; the ABI row is `abi2`; the others are as named.

| inputs | key | same key as the base |
|---|---|---:|
| the same inputs again | 55f983051b9d | yes |
| one comment line appended to the source | 006b1a586e51 | no |
| another entry name | 1de1c9327c73 | no |
| another device architecture, sm_70 | 23c739eafaf4 | no |
| another compiler version, nvrtc 13.0 | 03454a2c636f | no |
| a flag changed, --fmad=true | 15fad8851391 | no |
| a specialization macro, -DTILE=16 | c4afad16c679 | no |
| another launch ABI version | f83e576cb7bc | no |

An ahead-of-time image is looked up by a key without the compiler version (no compiler is present where it is loaded; the builder's version is recorded beside the image): the fields source, entry, architecture, flags and ABI, in this order (five fields, each length-prefixed as above), give the key e9d87cd78c86. Their values: the same source text, `brain_scale_shift`, `sm_61`, the flag string `--gpu-architecture=sm_61 --fmad=false` (the architecture option is part of the flags of an AOT build) and `abi1`.

Why each field is length-prefixed: without the prefix two different splits of the same characters hash alike.

| inputs | digest without lengths | digest with lengths |
|---|---:|---:|
| fields ('ab', 'c') | ba7816bf8f01 | 43ee655579de |
| fields ('a', 'bc') | ba7816bf8f01 | 9a8acca1b6c6 |
| same key | yes | no |
<!-- /worked -->

The key is the hash of the inputs, each preceded by its length. The second table shows why: with plain concatenation the fields ("ab", "c") and ("a", "bc") hash the same, and with lengths they do not. The AOT key leaves out the compiler version. This is a decision, not an omission: the point of AOT is that the machine that loads the image has no compiler to ask, so the version cannot be part of what is looked up; it is recorded beside the image for people instead, and the five fields that remain keep a stale image from being served for edited source. Publishing is part of correctness too: a cache file is written under a temporary name and renamed into place, so that a process that starts at the same moment sees the old entry or the complete new one and never half of a file.

**What an AOT build covers.** An ahead-of-time build chooses a list of architectures, and a device gets the image by a rule. Take the rule of a loader that tries, in order, the image for the device's own architecture, the images for lower minor versions of the same major (an image is binary-compatible upward within a major version; whether the driver agrees is the driver's decision, which is why the loader tries each candidate), and a portable form that the driver itself compiles at load time if its virtual architecture is not above the device's. A build for an older toolkit lane ships six architectures and one portable form; a build for a newer lane ships three and a portable form for its oldest:

<!-- worked: backends.aot_coverage -->
Which prebuilt image a device gets, by the loader's rule: its own architecture, else a lower minor of the same major, else portable code whose virtual architecture is not above the device's. Cubins of the older toolkit lane: sm_61, sm_70, sm_80, sm_86, sm_89, sm_90 and portable compute_61; of the newer lane: sm_90, sm_100, sm_120 and portable compute_90. 'none' means a run needs the compiler at the device. Whether the driver accepts a lower-minor image is the driver's decision; the rule is the loader's order of attempts.

| device compute capability | toolkit 12 images | toolkit 13 images |
|---|---|---|
| 6.0 | none | none |
| 6.1 | cubin sm_61 | none |
| 7.0 | cubin sm_70 | none |
| 7.5 | cubin sm_70 (lower minor) | none |
| 8.0 | cubin sm_80 | none |
| 8.6 | cubin sm_86 | none |
| 8.7 | cubin sm_86 (lower minor) | none |
| 8.9 | cubin sm_89 | none |
| 9.0 | cubin sm_90 | cubin sm_90 |
| 10.0 | portable form, driver compiles (compute_61) | cubin sm_100 |
| 10.3 | portable form, driver compiles (compute_61) | cubin sm_100 (lower minor) |
| 12.0 | portable form, driver compiles (compute_61) | cubin sm_120 |
| 12.1 | portable form, driver compiles (compute_61) | cubin sm_120 (lower minor) |
<!-- /worked -->

The two lanes cover different devices. The older lane serves every capability from 6.1 to 9.0 with a real image and the newer ones through the portable form; it serves 6.0 not at all, because the portable form needs an architecture at or below the device's, and 6.0 is below 6.1. The newer lane serves 9.0 and above with real images and nothing below 9.0, because it has dropped those architectures. A device left with `none` needs a compiler at the machine or another build. This is the trade of the design in one table: the size of the directory (3,900,000 bytes in the model for the six-architecture lane, 2,100,000 for the three-architecture lane) is bought with coverage, and the portable form is a cheap middle that costs a driver compile at load time and no toolkit.

## Same results: what "same" can mean

Three devices that compute the same function in float32 do not, in general, return the same bits, so a test has to say what it demands, and the tolerance has to come from the arithmetic and not from a guess.

**Cross-backend parity** is the property that every backend computes, for the same inputs, results that agree to a stated criterion. There are three criteria, in order of strength.

*Bit-identical*: every output word is the same on every backend. This is achievable, and the strongest test, when every rounding and every order of operations is fixed by the kernel and no backend changes either. A kernel that only moves values (a transpose), or that adds in an order the kernel itself spells out and whose backends all round each operation separately, is bit-identical everywhere.

*Within a tolerance derived from the operation*: each output differs from a reference by at most an amount that follows from the error bound of the arithmetic. This is what the rest of the cases need.

*The same decision*: the greedy token or the argmax is the same, a statement about a function of the outputs. It is the property a user cares about, and the weakest, because two logits within one tolerance of each other may swap.

**Why the bits differ.** Three reasons, each already met in an earlier chapter, now as a cause of disagreement between devices.

*Contraction.* Chapter 8 showed that a compiler may fuse $x y + z$ into one instruction that rounds once where the kernel's two operations round twice. Here is one triple, found by drawing seeded triples until the two results differ:

<!-- worked: backends.fma_example -->
The first triple whose two evaluations differ. Generator: the book's SplitMix64, seed 0xb2b605d084cfcbd1 (`seed_of` of the name `backends.fma_example`, defined on the pseudocode page). Draws in order: for each triple x, y, z, each `2 uniform() - 1` rounded to float32; the triple is number 6 drawn. The value column prints each float32 number exactly as the double-precision number that holds it (its shortest identifying decimal); 'exact' rows are the exact rational values to 22 significant digits.

| quantity | value | bits |
|---|---|---|
| x | 0.2752675712108612 | 0x3e8cefdf |
| y | 0.3516843020915985 | 0x3eb40ff7 |
| z | 0.16301369667053223 | 0x3e26ed10 |
| x y, exact | 0.09680728366974111764875 | - |
| f32(x y), the product rounded | 0.09680728614330292 | 0x3dc642e6 |
| x y + z, exact | 0.2598209803402733442113 | - |
| two roundings: f32(f32(x y) + z) | 0.25982099771499634 | 0x3e850742 |
| one rounding: f32(x y + z) | 0.25982096791267395 | 0x3e850741 |

The two results are 1 float32 step apart.
<!-- /worked -->

The product $x y$ needs more than 24 bits; rounding it before the addition throws away a part that the single rounding keeps, and the two results are one float32 step apart. A device that contracts and one that does not are both correct, and they are not equal.

*Order.* Chapter 7 showed that a left fold, a tree and a two-pass reduction round differently and give different bits. The same dot product of $n$ products, run as three programs on seeded inputs: the left fold with each product rounded (L), the left fold with a fused multiply-add per term (F, a device that contracts) and the tree of adjacent pairs (T):

<!-- worked: backends.dot_parity -->
The dot product of two vectors of n float32 numbers in [-1, 1), 40 seeded pairs per length n. Generator: the book's SplitMix64, seed 0x6dce362d4bc058f5 (`seed_of` of the name `backends.dot_parity`, defined on the pseudocode page). Draws in order, pair after pair, lengths 16, 64, 256, 1024, 4096 in turn: the n numbers of x (each `2 uniform() - 1` rounded to float32), then the n of y. Three float32 programs: L, the left fold (each product rounded, then added in order); F, the left fold with a fused multiply-add per term (one rounding per term); T, each product rounded, then an adjacent-pair tree. The exact value is the sum of the stored products, each exact in double precision, added exactly (`math.fsum`). S = sum |x_i y_i| (mean over the pairs). Errors are |computed - exact| / (u S), u = 2^-24, the largest of the 40 pairs; 'bound' is k / (1 - k u), the bound k u S / (1 - k u) in units of u S, with k the roundings on the longest path of the program (n for L and F; 1 + ceil(log2 n) for T).

| n | S | L: largest error | L: bound | F: largest error | F: bound | T: largest error | T: bound |
|---|---:|---:|---:|---:|---:|---:|---:|
| 16 | 3.84 | 1.22 | 16 | 0.804 | 16 | 0.5 | 5 |
| 64 | 16 | 1.62 | 64 | 1.76 | 64 | 0.757 | 7 |
| 256 | 64.7 | 1.37 | 256 | 1.37 | 256 | 0.203 | 9 |
| 1,024 | 255 | 0.866 | 1.02e+03 | 1.06 | 1.02e+03 | 0.126 | 11 |
| 4,096 | 1.03e+03 | 1.22 | 4.1e+03 | 1.29 | 4.1e+03 | 0.0966 | 13 |

Agreement of the programs with one another on the same inputs. 'Pair bound' is the sum of the two bounds of L and T; the last column asks whether the observed differences stay within the sum of the two programs' bounds.

| n | L = F bitwise | L = T bitwise | F = T bitwise | largest |L - F| / (u S) | largest |L - T| / (u S) | pair bound (L, T) | within the bounds |
|---|---:|---:|---:|---:|---:|---:|---:|
| 16 | 18 of 40 | 14 of 40 | 13 of 40 | 1.08 | 1.08 | 21 | yes |
| 64 | 9 of 40 | 5 of 40 | 6 of 40 | 0.802 | 1.47 | 71 | yes |
| 256 | 8 of 40 | 2 of 40 | 5 of 40 | 0.512 | 1.17 | 265 | yes |
| 1,024 | 7 of 40 | 4 of 40 | 5 of 40 | 0.335 | 0.992 | 1.04e+03 | yes |
| 4,096 | 4 of 40 | 0 of 40 | 2 of 40 | 0.281 | 1.26 | 4.11e+03 | yes |
<!-- /worked -->

L and F agree on every bit for 18 of the 40 vectors at $n = 16$ and for 4 of 40 at $n = 4{,}096$; L and T agree on none of 40 at the longest length. And every comparison obeys the bounds. The error of each program against the exact sum is far below its bound: the left fold's largest error is 1.22 $u S$ at $n = 4{,}096$ against a bound of 4,100 $u S$, the tree's is 0.097 against 13. The bound is a worst case, reached only if every rounding errs in the same direction. What matters for a test is that it can *never be exceeded* by a correct program, which is the property a tolerance needs, and how much of a defect it leaves invisible, which is the next paragraph.

![Error of a float32 dot product in units of u S against the number of terms n from 16 to 4,096, on logarithmic axes. The bound of the left fold rises with slope 1 to about 4,100; the bound of the tree rises slowly to 13. The largest errors of the left fold and of the tree and the largest difference between the two programs, over 40 seeded vectors per length, all stay below 2, and the tree's falls below 0.1.](img/parity-bound.svg)

*Library functions.* A function like $\exp$ is implemented by the device, and two implementations of the same function are both right to within their own accuracy. Here a reference rounds the double-precision library $\exp(x)$ and a "fast" path goes through a base-2 instruction, $2^{x \log_2 e}$ with the product rounded to float32:

<!-- worked: backends.exp_parity -->
exp(x) for 4000 seeded x. Generator: the book's SplitMix64, seed 0x8de567958b98efb8 (`seed_of` of the name `backends.exp_parity`, defined on the pseudocode page). Draws in order: x_i = `40 uniform() - 20` rounded to float32. The reference rounds the double-precision library exp(x) to float32. The 'fast' path computes t = x L in float32 with L = float32(log2 e) = 1.4426950216293335, then 2^t in double precision rounded to float32 (the shape of a device exp built on a base-2 instruction). Relative difference |fast - ref| / ref in units of u; 'ulps' is the distance in float32 steps; the bound is (2 |x| + 2) u at the upper end of the band: t carries a relative error of about 2 u (one rounding of L, one of the product), which 2^t multiplies by ln 2 |t| = |x|, plus one final rounding on each side.

| |x| in | values | bitwise equal | largest ulps | mean difference / u | largest difference / u | bound / u |
|---|---:|---:|---:|---:|---:|---:|
| 0 to 1 | 196 | 159 | 1 | 0.248 | 1.91 | 4 |
| 1 to 4 | 609 | 250 | 4 | 1.01 | 4.24 | 10 |
| 4 to 8 | 819 | 164 | 7 | 2.36 | 8.2 | 18 |
| 8 to 16 | 1598 | 145 | 14 | 4.83 | 15.2 | 34 |
| 16 to 20 | 778 | 54 | 14 | 6.3 | 16.1 | 42 |
<!-- /worked -->

The difference grows with $|x|$: the product $x \log_2 e$ carries a relative error of about $2u$, and the exponential multiplies a relative error in its argument by its relative condition number $\kappa_f(x) = |x f'(x) / f(x)| = |x|$ (Volume I, Chapter 15), so the result carries about $2|x|\,u$ on top of the final roundings. The bound of the last column, $(2|x| + 2)\,u$, is that argument; the observed difference lies below it in every band. A tolerance for a transcendental is therefore derived from its condition number, and a single absolute number for a kernel that applies $\exp$ to a wide range is wrong at one end or the other.

**Deriving a tolerance.** The recipe follows from the three cases. For a result built by a chain of operations, the largest number of roundings $k_{\mathrm{rd}}$ on any path from an input to the output gives the bound of Volume I, Chapter 15,
$$|\text{computed} - \text{exact}| \le \frac{k_{\mathrm{rd}}\,u}{1 - k_{\mathrm{rd}}\,u}\,S, \qquad S = \sum_i |x_i y_i|,$$
for a dot product of $n$ terms: $k_{\mathrm{rd}} = n$ for a left fold (the first product is rounded once and meets all $n - 1$ additions, $1 + (n - 1)$ roundings; every later term meets fewer) and $1 + \lceil \log_2 n \rceil$ for a tree (every path from a product to the root has one product rounding at the leaf and one addition rounding at each of the $\lceil \log_2 n \rceil$ levels). A backend is held to this bound against the *exact* value, each backend by its own $k_{\mathrm{rd}}$, and two backends agree if both do; two backends then differ by at most the sum of their two bounds. The tolerance scales with $S$, the size of the terms, and not with the size of the answer, because a sum of large terms that cancel is exactly the case where the answer is small and the error is not.

A tolerance that is chosen by looking at what passes is wrong in two directions, and the same seeded vectors show both. The device under test is the tree program T, and the defect is the under-dispatch of Chapter 5: the last product is never added. Four tolerances are tried against the correct device and against the defective one:

<!-- worked: backends.defect_detection -->
The device under test runs program T of `backends.dot_parity` on the same 40 seeded pairs. The defect is the under-dispatch of Chapter 5: the last product x_(n-1) y_(n-1) is never added (replaced by 0 in the tree). A test compares the device's result with the exact value and flags it when the difference exceeds its tolerance; the tolerances are a fixed absolute 1e-6, 1e-3 S, and two derived ones: k u S / (1 - k u) for the tree (k = 1 + ceil(log2 n)) and for a left fold (k = n). 'False alarms' count correct results flagged; 'detected' counts defective results flagged.

| n | tolerance | false alarms (correct device) | defect detected |
|---|---|---:|---:|
| 1,024 | fixed 1e-6 | 12 of 40 | 40 of 40 |
| 1,024 | loose, 1e-3 S | 0 of 40 | 18 of 40 |
| 1,024 | derived, tree, k = 1 + ceil(log2 n) | 0 of 40 | 40 of 40 |
| 1,024 | derived, left fold, k = n | 0 of 40 | 40 of 40 |
| 4,096 | fixed 1e-6 | 24 of 40 | 40 of 40 |
| 4,096 | loose, 1e-3 S | 0 of 40 | 0 of 40 |
| 4,096 | derived, tree, k = 1 + ceil(log2 n) | 0 of 40 | 40 of 40 |
| 4,096 | derived, left fold, k = n | 0 of 40 | 18 of 40 |
<!-- /worked -->

The fixed absolute $10^{-6}$ is the number that suits unit-sized terms at a tiny shape, and it flags the *correct* device on 12 of 40 vectors at $n = 1{,}024$ and 24 of 40 at $n = 4{,}096$: the correct tree has errors of about $u S$, and $S$ grows with $n$. The loose relative tolerance $10^{-3} S$ never flags the correct device and misses the defect: it detects 18 of 40 at $n = 1{,}024$ and none at 4,096, because a dropped term of size about $\tfrac14$ is smaller than the tolerance. The tolerance derived for the tree, $k_{\mathrm{rd}} u S/(1 - k_{\mathrm{rd}} u)$ with $k_{\mathrm{rd}} = 1 + \lceil \log_2 n \rceil$, has no false alarm and detects the defect on 40 of 40 at both lengths. The tolerance derived for a left fold (which is $k_{\mathrm{rd}} = n$, and is the right one for a backend that folds) detects 40 of 40 at $n = 1{,}024$ and only 18 of 40 at $n = 4{,}096$: its bound has grown to 0.25, as large as the term that went missing. Two consequences. The tolerance a test may use is the one of the program that runs, so a backend whose order is fixed to a tree can be held to the tight bound; and a test that cannot tell the defect it exists to find from the rounding of a long left fold has no teeth, which is the reason to fix the order when the stakes require it (as Chapter 7 did for a reduction that must be reproducible).

**A parity test, designed.** The pieces assemble into a procedure:

1. *Inputs.* Seeded, with the seed printed; sizes that are not multiples of the workgroup size (so the tails exist), and at least one case whose terms nearly cancel.
2. *Reference.* The exact value (or a double-precision evaluation), or a reference backend whose order every other backend is built to reproduce.
3. *Criterion per kernel.* Bit-identical where the kernel fixes every rounding and the backends are forbidden to contract; otherwise a tolerance derived from $k_{\mathrm{rd}}$ and $S$ (or from $\kappa_f$ for a library function), computed, never tuned.
4. *Teeth.* For each mechanism the backend implements (the clamp, the zeroing, the grid, a sub-range offset), remove it and check that the test fails. A test that cannot fail is not evidence (knowledge #116).
5. *Absence is not a pass.* A device that is missing must be reported as skipped; a green run that tested nothing is worse than no test.

The **parity matrix** is the table of the kernels against the backends, each cell the result of such a test. Here five kernels against three modelled arithmetic profiles, on seeded inputs: P0 is the reference (every operation rounds in the order the kernel states; this is what a CPU loop nest and a translator that forbids contraction compute), P1 contracts a product followed by an add where the kernel has one, and P2 computes $\exp$ through the base-2 instruction. These are models of behaviours, not measurements of a vendor's device:

<!-- worked: backends.parity_matrix -->
Three modelled arithmetic profiles on the same seeded inputs. Generator: the book's SplitMix64, seed 0x5aa1acba3a2a6429 (`seed_of` of the name `backends.parity_matrix`, defined on the pseudocode page). Draws in order: 4096 x for y = a x + b (`2 uniform() - 1` rounded to float32; a = 1.7 and b = -0.3 rounded to float32); 4096 slot values (`2 uniform() - 1` rounded to float32; output i is (s[4i] + s[4i+1]) + (s[4i+2] + s[4i+3]), each addition rounded); 4096 pa then 4096 pb (the same distribution) for the dot products (output i uses pa[64i .. 64i+63] and pb[64i .. 64i+63], with the tree of `backends.dot_parity`: each product rounded, then adjacent pairs added); 4096 x for exp (`16 uniform() - 8`, rounded to float32). The transpose draws nothing: its input is the [6, 10] matrix with entry 10 i + j in row i, column j. Profile P0 (the reference, and what a CPU loop nest and a translator that forbids contraction compute): every operation rounds to float32 in the order the kernel states, exp is the double-precision library value rounded to float32. Profile P1 (a device that contracts a product followed by an add into a fused multiply-add where the kernel has one: for y = a x + b the fma(a, x, b); in the dot-product tree only the first level, where leaf j = fma(pa[2j+1], pb[2j+1], f32(pa[2j] pb[2j])) fuses the odd element's product into the add of the even product, and the remaining levels add as in P0). Profile P2 (a device whose exp goes through a base-2 instruction, as in `backends.exp_parity`). Each cell compares a profile with P0 on bit patterns; where values differ it also prints the largest difference in units of u S, with S = |a x| + |b| for y = a x + b, the sum of the absolute values of the 64 products for a dot product and |exp(x)| for exp.

| kernel | P0 (reference) | P1: contracts | P2: fast exp |
|---|---|---|---|
| y = a x + b, 4096 values | identical (4096 of 4096) | 1560 of 4096 differ, at most 72 ulp, at most 1.99 u S | not applicable |
| transpose of a [6, 10] matrix | identical (60 of 60) | identical (60 of 60) | not applicable |
| sum of 4 slots, 1024 outputs | identical (1024 of 1024) | identical (1024 of 1024) | not applicable |
| dot product of 64 terms, tree, 64 outputs | identical (64 of 64) | 35 of 64 differ, at most 192 ulp, at most 0.63 u S | not applicable |
| exp(x), x in [-8, 8), 4096 values | identical (4096 of 4096) | not applicable | 2636 of 4096 differ, at most 7 ulp, at most 7.95 u S |
<!-- /worked -->

The pattern of the cells is the lesson. The transpose and the four-slot sum are bit-identical on every profile, because the first has no arithmetic and the second only additions in a fixed order: for such kernels bit-identity is the right criterion, and a tolerance would only hide a defect. The kernels that have a multiply followed by an add differ on a large fraction of their outputs under contraction (1,560 of 4,096 for $a x + b$, 35 of 64 dot products) by at most $2\,u S$ (1.99 and 0.63 in the table), which the bound for two roundings predicts; the largest differences measured in float32 steps (72 and 192) look alarming and are not, because the results nearly cancel and the step of a small number is small, which is why the units of a tolerance are $u S$ and not ulps. And the $\exp$ kernel differs in most of its outputs, by at most about $8u$ over $|x| < 8$.

## The backend seam, the two compilers and the parity tests in brain's crates

**The seam.** `crates/backend-api/src/lib.rs` holds the contract. Its `Backend` trait is the interface described above: allocate (`storage`, `buffer`, `storage_init`, `uniform_dynamic`), upload (`write`, `write_at`), record a dispatch (`step`, `step_sliced`, `step_buf`; a kernel is named by its index in the list given at construction), run it (`submit`), `read` (which blocks), `poll_wait`, and `caps`, with a `kind()` name (`"wgpu"`, `"cpu"`, `"vulkan"`, `"cuda"`) for diagnostics. Handles are opaque: `DeviceBuffer` and `Step` wrap `Arc<dyn Any>` and each backend downcasts its own. A backend registers a factory by name (`register_backend`, `create_backend`), which is what makes a new backend a crate that depends on `backend-api` and nothing else. Commit 7c8c97045 (July 2026) introduced the seam, moving the wgpu, CPU and Vulkan backends into crates of their own behind a facade, with the existing public API unchanged. It introduced a second contract for whole-graph accelerators too, which the crate's header now records as deleted: it was never object-safe, had one implementation and one call site, and could not express named multi-tensor input, so an interface with one user was not worth keeping. The record `DeviceCaps` carries what a selector needs (`max_workgroup_size`, `workgroup_mem_bytes`, `subgroup_size` as an `Option`, `workgroup_reductions`, and `peak_gflops` and `peak_bandwidth_gbs`, which are filled by measurement and stay `None` until something measures them). Its documentation states the rule above: an unknown capability is never assumed present, and `workgroup_reductions` is "a correctness gate, not a tuning preference". Knowledge #105 records that the selection policy reads `DeviceCaps` and never a backend name; its lesson is the other half of the abstraction, that two kernels a capability chooses between are interchangeable only if they read the same parameters, and a kernel-by-kernel comparison cannot see a model that registered the wrong pair.

**The three GPU and CPU routes.** `crates/backend-wgpu/src/lib.rs` compiles every kernel of the list when it is constructed (a shader module and a compute pipeline per kernel, `create_shader_module_trusted`, runtime checks off unless the environment variable `BRAIN_GPU_CHECKED` is set) and keeps a driver pipeline cache per adapter on disk, keyed by the adapter and driver identity (`PlCache`), and ignores a stale or corrupt blob. `crates/backend-vulkan/src/lib.rs` is the native route: `crates/vulkan/src/shader.rs` compiles the same WGSL to SPIR-V (`wgsl_to_spirv`) and reads the bindings back from the source to build the descriptor layout; its pipeline cache (`crates/vulkan/src/pipeline_cache.rs`) is keyed by vendor, device and the driver's cache UUID, so a driver update invalidates it by filename. Commit fffeba448 (June 2026) added it as a runtime-selected third backend. Both read the same `BRAIN_GPU_CHECKED` switch, because knowledge #51 found two GPU backends compiling one WGSL under different rules. `crates/backend-cpu/src/lib.rs` keeps a buffer as host memory, runs the recorded steps in order and spreads the invocations of each step over a thread pool in disjoint ranges aligned to the workgroup size; its capability record sets `workgroup_reductions` to false.

**The CPU compiler.** `crates/wgsl-cpu/src/lib.rs` parses with the same front end, translates its IR to the code generator Cranelift and JITs one native function per kernel. Commit 071774828 (June 2026) added it for kernels without barriers or atomics, each invocation owning one output; commit 00f7cece6 (June 2026) added the workgroup model described above, in the shape the comment on `compile_one_wg` gives (`for wg: for lid: <before>; for lid: <after>`), with `var<workgroup>` as a stack array and the body split by `split_at_barrier`. The code carries the three mechanisms of the pseudocode: `wg_fill` clears every workgroup array at the start of each workgroup, the "still running" mask in workgroup scratch carries a return across the split, and `emit_invocation_loop` is the loop. Commit 3f489e789 (September 2026) added the last two after finding both by comparing this compiler with a second one over the same WGSL (knowledge #112); the tests are `an_invocation_that_returned_before_the_barrier_stays_returned` and `workgroup_memory_is_zero_at_the_start_of_every_workgroup` in `crates/wgsl-cpu/tests/it/workgroup_semantics.rs`.

The compiler's story of locals is the cautionary tale. Commit 9b9c9c003 (August 2026) fixed a kernel whose per-token activation maximum on the CPU backend returned row 0's answer for every workgroup after the first. The compiler re-materializes the values computed before the barrier at the top of the second loop, and it had also done so at the top of the first, where a value that reads a local variable assigned earlier in that same segment was evaluated before the assignment ran, read the zero the variable starts with, and cached it. The commit's minimal reproduction needed no reduction and no second workgroup: `var a = x[row]; partial[t] = a;` was enough. Commit 77f96d773 (August 2026) then enforced the invariant that the compiler's own comment had claimed and nothing checked: a local that is stored before the barrier and loaded after it is now a compile error (`locals_touched` in the code), the refusal rather than the promotion of the second table above, pinned by tests in `crates/wgsl-cpu/tests/it/workgroup_locals.rs`. A compile failure is a typed value, `CompileError`, and the compiler decides by its variant, never by what the message says, whether a failure means "this kernel's structure is not expressible, run it elsewhere" (`UnsupportedWorkgroupStructure`) or "this kernel or the compiler is broken" (`Failed`); a genuine miscompile must not be swallowed as a skip, and a test names a local after a barrier to pin that a message mentioning one decides nothing. A kernel whose barrier structure the compiler does not support is skipped with a message and a dispatch of it panics (`Jit::run`); the code's comment says every such kernel declares `@cpu no` or `native-only` in its header. `crates/wgsl-cpu/tests/it/compile_all.rs` builds the compiler over every kernel of the catalogue, so a genuine compile error fails it, and a second test there allows exactly one documented exception for the barrier structure.

**The CUDA translator.** `crates/wgsl-cuda/src/lib.rs` is the translator of the first listing. Its header lists six hazards, each handled and each with a test in `crates/backend-cuda/tests/it/wgsl_cuda_golden.rs` (for the sixth, `__restrict__`, the test asserts only that the keyword is never emitted; the commit notes that no covered kernel could miscompile without it): a return before a barrier becomes a flag (`__active`, "never a return"), uniform members are read at the byte offsets of the WGSL layout, shift counts are masked, contraction is forbidden (the compiler runs with `--fmad=false`, set in `crates/backend-cuda/src/nvrtc.rs`, and the emitter uses `1.0f / sqrtf(x)` and not the fast reciprocal square root, and `rintf` for WGSL's round-half-to-even, as knowledge #114 records), no `__restrict__` is ever promised, and workgroup memory is zeroed and published by a barrier at entry. Signed integer arithmetic goes through the unsigned type, f16 is refused rather than run as float32, and everything outside the supported IR is an error that names the construct. Commit 3101d39c5 (September 2026) introduced the translator for six kernels chosen to reach every mechanism and says that its tests were checked to fail when the handling was removed; the test for contraction searches a seeded stream for triples on which the two roundings differ and asserts that it found some, so that it cannot pass vacuously. Commit b3d5cf301 (October 2026) added the clamp after the flaky test of knowledge #202. Commit 8b23e0857 (October 2026) refined the guard transformation with an analysis of which returns and barriers are uniform across the workgroup (`crates/wgsl-cuda/src/uniform.rs`): a return that the whole workgroup takes needs no guard, a barrier under uniform control flow is emitted where it stands, one under non-uniform flow is refused with that reason, and a return taken by part of the workgroup keeps the guard.

**How agreement is tested.** The tests have three layers. `crates/backend-cuda/tests/it/wgsl_cuda_golden.rs` runs one WGSL source through two independent code generators, the CPU compiler and the CUDA translator through NVRTC, per kernel and per shape; the tolerance of each case is in its `outputs`: 0.0 (bit-identical) for the elementwise add and multiply and the packed-int8 kernel, and, for the tanh-based GELU (a library call on one side, a device intrinsic on the other), the cooperative reduction and the arctangent, a bound computed from the case itself. It is the round-off bound of the dot-product tables above, $\gamma_k S = k u S/(1 - k u)$ with $u = 2^{-24}$, $k$ the number of rounded operations on the longest chain and $S$ the sum of the absolute values of the terms, which `brain_testutil::rounding` (`crates/testutil/src/rounding.rs`) implements once and every case calls with its own $k$ and $S$: for a workgroup's partial sum of squares, $k$ is one product, the strided adds of a lane and the 64 adds that fold the lanes, and $S$ is the partial sum itself; for the GELU, a chain of nine operations plus the two ulp each `tanh` is allowed. Two evaluations that are each within the bound of the exact value are within twice of each other, which is the tolerance between backends. `crates/gpt2/tests/it/cuda_backend_parity.rs` runs a whole decoder forward on every backend present and holds each logit to a bound of the same form, with $k$ counting every reduction from the token to the logit and $S$ the absolute sum of the head's terms, and the argmax ids to equality. Knowledge #116 measured how little the second says about plumbing: with five changes applied one at a time (storage not zeroed, a sub-range binding ignored, the grid laid out at a fixed 64, an offset dropped, contraction allowed) that model test still passed, so `crates/backend-cuda/tests/it/backend_contract.rs` gives each mechanism its own assertion. A bound that follows the length and the scale of each case is what the tables above ask for: a fixed one would flag a correct device at a longer reduction and hide a defect at a short one. Where identity is achievable, brain asks for it, and says so: `add_rms_quant.cu` of Chapter 8 keeps the order of the portable kernels so its gate compares bytes. `scripts/gates/parity-gate.sh` runs the gradient-check suite on each backend in turn (CPU, Vulkan, CUDA) and compares a native kernel with its portable reference on the same device.

**JIT and AOT in the CUDA backend.** `crates/backend-cuda/src/backend.rs` compiles lazily: registering a catalogue compiles nothing and a kernel is translated and compiled at its first dispatch (`compiled_for`; the test `construction_compiles_no_kernel` pins it), the lazy row of the table above. `crates/backend-cuda/src/nvrtc.rs` holds the JIT cache: the key (`cache_key`) is the SHA-256 of the length-prefixed source, entry name, target, NVRTC version, flags, specialization macros and a launch ABI version, and a unit test in the same file varies each input in turn and requires the key to change. `crates/backend-cuda/src/aot.rs` and the tool `crates/cuda-aot/src/main.rs` (`make cuda/aot`) are the ahead-of-time half: the tool builds every kernel of the catalogue through the translator and the native kernels for each architecture of a toolkit lane (`Plan::for_toolkit`) and writes a tab-separated manifest keyed as in the model (`aot_key`, without the compiler version), with the SHA-256 of each image checked before it reaches the driver. If the translator refuses a kernel, the tool fails and writes no directory, because a directory that quietly lacks a kernel would hide the defect until a model needed it. The loader `compile_with` in `crates/backend-cuda/src/exec.rs` tries the images in the order of the model (own architecture, lower minors, the disk cache or NVRTC, portable form) and reports every attempt when none works, naming `make cuda/aot` as the remedy. Commit 72806b62d (October 2026) introduced it for machines that have a driver and no toolkit; its message lists the architectures of the two lanes, the ones in the table.

## Check your understanding

1. Find the error: "WGSL to CUDA is a change of syntax. It compiles, and the elementwise kernel gives the same answers on a 1,024-element test, so the translation is correct."
2. A kernel computes `1u << s`. The reference gives 1 for $s = 32$ and a CUDA target gives 0. Which is correct, and what must the translation do?
3. A CPU compiler zeroes shared memory only at the start of the program. Every kernel in the catalogue writes every slot before reading it. Is the compiler safe, and what kernel would expose the defect?
4. A team holds a dot product of 4,096 terms to an absolute tolerance of $10^{-6}$ on a CPU and a Vulkan GPU, and the test fails on a correct device. Using the tables of this chapter, say why and what the tolerance should be.
5. A device of compute capability 8.9 starts a program built with the newer toolkit lane of `backends.aot_coverage`. What happens, and what are the two ways to fix it?
6. Two backends run the same reduction with different launch shapes (Chapter 7). A colleague asks for bit-identity between them. Can the test demand it, and what must change if the answer is yes?
7. Why may an AOT key leave out the compiler version while a JIT key may not, and why must both contain the flags?
8. Find the error: "Our tolerance is $k_{\mathrm{rd}} u S/(1 - k_{\mathrm{rd}} u)$ for the left fold, $n = 4{,}096$, so a test at that tolerance proves a backend that folds is correct."

## Summary

One kernel source reaches three kinds of device as three different programs: SPIR-V compiled by a driver, CUDA C++ compiled by NVRTC (or loaded as a cubin built beforehand), and a loop nest in machine code, and the model above them sees only a narrow backend abstraction: capabilities, compile, buffers, upload, dispatch, read-back, release, with opaque handles, an ordered queue, a read that waits, and capabilities in which an unknown value forbids the feature. Keeping one kernel set turns $K D$ files into $K + D$ artifacts without removing any comparison. Shader translation is correct when it keeps the promises that a target does not make by itself: clamped indices, wrapping integers, a chosen shift rule, rounding to nearest with no contraction, barriers that every invocation reaches, a return that is permanent, zeroed workgroup memory and the WGSL byte layout. On a CPU a workgroup becomes two loops split at the barrier (the simulator of Chapter 5 is this transformation), and the loops must clear the shared array per workgroup, carry a return across the split in a mask and promote locals that cross the barrier. JIT compilation pays the compiler on the machine at start (120,000,000 steps for a whole catalogue of the model against 14,400,000 for the 12 kernels a run uses) under a key that covers every input that can change the code; ahead-of-time compilation pays it before shipping, runs where no compiler is present, and trades 2,100,000 to 3,900,000 bytes and a coverage that depends on the architectures built.

Cross-backend parity means bit-identity where the kernel fixes every rounding and order, and otherwise a tolerance derived from the number of roundings $k_{\mathrm{rd}}$ and the size of the terms $S$, never chosen from what passes: a fixed $10^{-6}$ flagged correct devices on 24 of 40 seeded vectors, a loose $10^{-3} S$ missed a dropped term on every vector at 4,096 terms, and the derived tolerance of the tree did neither. The bits differ for three reasons, contraction, order and library functions, and the parity matrix shows which kernels escape them. All of this assumes the devices compute the same function; what it leaves open is which of several correct kernels to run on a device nobody tuned for, since they all agree and differ only in work. That is the question of the next chapter.

## References

- Nicholas J. Higham, "The accuracy of floating point summation", 1993.
- Nicholas J. Higham, "Accuracy and Stability of Numerical Algorithms", 2002.

## Exercises

1. **[Calculate]** A program registers $N_{\mathrm{ker}} = 250$ kernels and dispatches $N_{\mathrm{used}} = 20$; $s_{\mathrm{src}} = 3{,}000$ bytes, $s_{\mathrm{img}} = 8{,}000$ bytes, a portable form of 4,000 bytes, $c_{\mathrm{cmp}} = 400$ steps per byte. Find the compiler steps of the eager start with an empty cache and of the lazy start with an empty cache, the bytes read by the lazy warm start and by the AOT start, and the disk size of an AOT directory with 6 architectures and one with 3.
2. **[Calculate]** With the loader's rule of `backends.aot_coverage`, find the image that devices of capability 7.2, 6.2, 9.5, 11.0, 5.2 and 8.8 get from the older and from the newer toolkit lane.
3. **[Calculate]** A kernel over $N = 1{,}000{,}003$ elements with workgroups of 256 is compiled to a CPU loop nest. Find the workgroups, the number of times the body runs, the number that fail the guard, and, for a kernel with one barrier and a 256-slot shared array, the words the loop nest clears over the whole run.
4. **[Calculate]** A dot product of $2^{20}$ numbers is computed by 256 lanes, each a left fold over 4,096 products, then a tree of 8 levels. Find $k_{\mathrm{rd}}$ and the tolerance in absolute terms for $S = 262{,}144$, and compare it with a pure tree.
5. **[Calculate]** Using the bound $(2|x| + 2)\,u$ for the fast $\exp$ of `backends.exp_parity`, find the relative tolerance for $|x| \le 88$ and the absolute tolerance on $\exp(10)$.
6. **[Implement]** Reproduce `backends.three_paths`: write the three executions (the invocation model in a shuffled order, the CUDA-like model with clamped indices and a uniform block of 32-bit words, the loop nest), the three listings from one description of the kernel, and the table. Add $N = 13$ with workgroups of 8 and report the number of times the body ran in each.
7. **[Implement]** Reproduce `backends.cpu_phases`: write the reference semantics and the loop nest with the two switches (return scope and clearing). Add workgroups of 8 with $N = 21$, 4 launched workgroups of which 3 are real, and report `out` for all five variants.
8. **[Implement]** Reproduce `backends.cache_key`: the SHA-256 of length-prefixed fields, with the CUDA text of `backends.three_paths` as the source (the text ends with a newline). Report the first 12 hexadecimal digits for each row and add a row for a changed entry name that is one character longer.
9. **[Implement]** Reproduce `backends.dot_parity` and `backends.defect_detection` from the seeds and draw orders printed there, with float32 rounding after every operation, a fused multiply-add as one rounding of the exact value and an exact sum for the reference. Add a fourth program, the tree with a fused multiply-add at its leaves, and give its $k_{\mathrm{rd}}$ and its largest error.
10. **[Implement]** Reproduce `backends.exp_parity` and `backends.parity_matrix` (the double-precision library `exp` rounded to float32 is the reference). Add the kernel $\tanh$ for the same 4096 values of $x$ (the draws of the exp kernel): the reference is the library $\tanh$ in double precision rounded to float32; the fast path computes $t = \mathrm{f32}(\mathrm{f32}(2x)\,L)$ with $L$ of `backends.exp_parity`, $E = \mathrm{f32}(2^t)$ (double-precision power rounded once) and the result $\mathrm{f32}(\mathrm{f32}(E - 1)/\mathrm{f32}(E + 1))$. Derive a bound from the condition number of $\tanh$ and report where the difference from the reference leaves it.
11. **[Investigate]** Compile a kernel with $a b + c$ on your toolchain at its default flags and at the flags that forbid contraction, and read the generated instructions. Report whether your default contracts, and run the seeded triples of `backends.fma_example` on your device to find the first triple on which the results differ.
12. **[Investigate]** Run a kernel of your own on two backends you have (two devices, or a device and a CPU path), derive the tolerance from its error bound as in this chapter, report the largest difference in units of $u S$ and whether it is within the tolerance, and then remove one mechanism (the guard, a clamp) and show that your test fails.

## Key terms

- **backend abstraction**: the narrow set of operations (capabilities, compile, create a buffer, upload, dispatch, read back, release) that every device-specific backend provides with the same meaning, so that the code that builds a model never names the device that runs it. <!-- concept: backend-abstraction -->
- **shader translation**: rewriting a kernel from its own language into the form a device accepts so that it still computes what the source computes, which requires keeping every promise the language makes (clamped bounds, wrapping integers, barrier and return semantics, zeroed workgroup memory, layout) that the target does not make by itself. <!-- concept: shader-translation -->
- **JIT compilation**: compiling a kernel to machine code at the moment a program needs it, on the machine that runs it, so that it fits the device present; the work of the compiler is paid at run time, usually saved in a cache. <!-- concept: jit-compilation -->
- **ahead-of-time compilation**: compiling kernels for a list of known device architectures before the program ships and loading the results at run time with a key that identifies their inputs, so that a machine without a compiler can run them, at the price of storage and of devices outside the list. <!-- concept: ahead-of-time-compilation -->
- **cross-backend parity**: the property that every backend gives results for the same inputs that agree to a stated criterion, bit for bit where the kernel fixes every rounding and order, and otherwise within a tolerance derived from the error bound of the operation. <!-- concept: cross-backend-parity -->

## Answers

**Recall questions.**

1. $3 \cdot 4 = 12$ invocations. The two with $g = 10$ and $11$ have no element; the guard makes them return before they read or write.
2. When every invocation of the workgroup takes it, so none is left to arrive. If some return and others reach the barrier, the others wait for invocations that never come and the result is undefined.
3. The sums are taken in different orders, and float addition is not associative, so the roundings differ (five lane counts gave five different results in Chapter 2). The eight-lane sum has the smaller bound, because each chain is shorter.
4. It forms $x y + z$ exactly and rounds once where the two operations round twice. 928 of 4,096 seeded triples (22.7 percent) differed.
5. It closes the list and hands all of its launches to the device as one submission, in order. A list that is still open has not been handed over, so there is nothing for the device to finish; a program reads back only after it has submitted what the result depends on.
6. $u = 2^{-24}$; the tree's bound is $\lceil \log_2 n \rceil\,u \sum_i |x_i|$.

**Check your understanding**

1. Five of the six translation hazards pass a happy-path comparison (the header of the golden test of the CUDA translator says so), and a whole-model parity test stayed green with each of five changes applied in turn (knowledge #116). The clamp matters only past the end of an array, the shift only for counts of 32 and more, the zeroing and the return only at a ragged edge. The translation is correct only if each promise has a test that fails when it is removed.
2. Neither: WGSL calls the range indeterminate. The translation must write the mask `(s & 31u)` so that every tier computes the reference's answer (1 for $s = 32$), as in the first table of `backends.preserved_semantics`.
3. Not safe. The defect shows with a kernel that reads a slot that some invocations never write because their `if g < N` is false: the sum kernel `wg_sum` of the chapter does, and the last real workgroup then adds the stale 7 and 8 of the previous workgroup (34 instead of 19). The reductions of an engine at a ragged end do the same.
4. The correct tree has an error of about $u S$ and $S \approx 1{,}030$ at $n = 4{,}096$, so about $6 \times 10^{-5}$, far above $10^{-6}$; the table shows 24 of 40 correct results flagged. The tolerance should be derived from the program: $k_{\mathrm{rd}} u S/(1 - k_{\mathrm{rd}} u)$ with $k_{\mathrm{rd}} = 13$ for a tree, about $13 u S = 8 \times 10^{-4}$ on average over these vectors (and it detected the dropped term in all 40).
5. The newer lane has images for 9.0, 10.0 and 12.0, nothing below 9.0, and its portable form is compute_90, above 8.9, so the device gets `none` and needs a compiler. Either build with the older lane too (it holds sm_89), or give the machine a toolkit so that it compiles at start.
6. Not unless the order is the same: launch shapes change the order of the additions (Chapter 7's three shapes gave +2, +3 and -3 steps), so the bits differ. To demand bit-identity the shape must be fixed by the data size alone (as brain's gradient-norm kernel does), or the test must use the tolerance of the program.
7. An AOT image is loaded where no compiler exists, so the version cannot be asked for and is recorded beside the image; a JIT key is formed where the compiler is, and a different compiler can emit different code. Both need the flags because `--fmad=false` against `--fmad=true` changes the rounding and so the numbers.
8. A test at that tolerance only proves that no result is wrong by more than the bound. At $n = 4{,}096$ the bound is 4,100 $u S$, about 0.25, as large as a dropped term, and the table shows 22 of 40 defective results passing. The test shows the arithmetic is within the worst case, not that it is correct; fixing the order allows the tighter bound.
