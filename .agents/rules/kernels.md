# Kernel checklist — read before writing, dispatching, or optimizing one

Every rule here was paid for with a real defect in this engine. This page is
the pre-flight; `docs/performance/overview.md` is number-free methodology
only (profiling, the runtime selector, INT8, `brain flops`) - the "why"
behind a specific case study, with real measured numbers, lives in that
model's own `.agents/roadmap/<model>.md`.

**Trigger**: you are about to add a `.wgsl`, dispatch an existing kernel from a
new model, or make something faster. Read the matching section first.

---

## A. Before you WRITE a kernel: does a good one already exist?

**The most expensive mistake in this repo is not a slow kernel — it is a fast
kernel nobody knew about.**

A slow reduction kernel was diagnosed and fixed for one model. A later crate was
written against the same kernel *name*, and silently inherited the slow one —
a multi-second decode step became 159× faster once it was found. Several
instances of one already-solved pattern were found in that single model.

Before adding a kernel:

1. `grep -rn '<op>' crates/kernels/wgsl/ | grep -v backward` — look for siblings
   with these suffixes, which mean "the fast one": `_rows` / `_wg` (cooperative,
   workgroup-per-row), `_reg` / `_reg2` / `_reg3` (register-tiled),
   `_tiled`, `_part` (two-pass reduction), `_dyn` (dynamic-quant int8).
2. If a fast sibling exists, **the fix belongs in kernel *selection*, not in a
   new copy.** Wire `backend_api::select` / `gpu_core::tune`, or resolve by name
   (`Gpu::kernel_index`) as `model::vit` does. A faster sibling that models must
   opt into by hand will be missed by the next model, exactly as above.
3. If you genuinely need a new variant, add it **alongside** and select per
   shape. Never silently retarget a kernel other models dispatch — `matmul_reg2`
   has half the repo behind it.
4. **A selection seam that still needs every dispatch site edited is a seam the
   next model will miss.** `backend_api::select` and by-name `Gpu::kernel_index`
   both leave the call site to map the answer to its own pipeline index, so
   adopting a fix is still N edits in N crates. Where the fast variant is a
   genuine **drop-in** — same `Params`, same bindings, *bit-identical* result,
   only a different thread count — put it in `gpu_core::upgrade` instead: `Gpu`
   appends it to whatever kernel list a model registers (at the end, so no index
   moves) and rewrites the *dispatch* in `step*`. One such upgrade reached three
   unrelated crates with **zero** changes in any of them — a >9x win in one
   model's int8 prefill. The bar is high on purpose — read that module's header
   before adding a row; a *sum* reduction fails the bit-identical rule and stays
   in the explicit seams, where a trajectory gate is visible at the call site.
   **Keep the recorded `StepMeta` in the caller's index space**: profilers map
   `meta.kernel` through their own kernel list, and an early version of this
   seam crashed a bench tool by handing it an appended slot. A row may also be
   **shape-specialised** - a `kernels::template` knob plus a bucket ladder, one
   appended pipeline per bucket, the dispatch picking from the caller's own
   uniform params - which is how a variant needing a compile-time constant
   (register accumulators) still reaches every model with zero edits. The four
   bars are unchanged, and "wins at every shape" is precisely what forces a
   ladder instead of one worst-case build (§C6). Six extra pipeline compiles
   measured as noise against device init, so the cost is not a reason to skimp
   on buckets.

## B. Before you DISPATCH an existing kernel: read its contract

Contract mismatches on kernels that were already correct have caused real bugs
more than once:

| defect | cost |
|---|---|
| `silu_mul` takes a **single `total`** param; passing `[rows, cols]` computed a small fraction of the MLP | forward cosine **0.504** — silently wrong, not a crash |
| `step_sliced` offsets/lengths are **f32 elements, not bytes** | SIGSEGV |
| `nchw_nlc`/`nlc_nchw`'s `hw` is **every axis below the channel**, not `H*W`. On a 5D `[C, T, H, W]` operand that is `T*H*W` | the Wan-VAE attention permuted the wrong axis. Invisible at `T == 1` - which is what every chunk of the reference's own encode AND decode feeds it, so no golden at any clip length could catch it. Found only by a bit-exact chunk-size-invariance test: cosine **0.99962** where the answer must be exactly 0.0 |

So, for every kernel you dispatch:

```bash
sed -n '/struct Params/,/^};/p' crates/kernels/wgsl/<k>.wgsl   # the contract
grep -rn 'K_<KERNEL>\|<kernel_name>' crates/*/src | head       # a working call site
```

Read the header comment (it states the layout and the invariants), match the
`Params` field order exactly, and copy the thread-count expression from an
existing caller. Assume nothing about units.

## C. Writing a kernel: the four traps that actually happened

1. **`var<function>` arrays and runtime-bounded loops become LOCAL memory.**
   One attention kernel held per-thread arrays indexed by a *runtime* head
   dimension: too big for the register file, and the runtime bound blocks
   unrolling, so both spilled to global-backed local memory — a tiny fraction
   of achievable throughput. It was the majority of that model's forward pass;
   a large multiple faster once fixed with lane-splitting and a compile-time
   trip count. *Sizes that index thread-private arrays must be compile-time
   constants (`kernels::template` specialises them).*

2. **One thread per row is a coalescing bug.** A single invocation walking a
   whole row serialises the loads and wastes most of every memory sector.
   Measured fixes to the cooperative workgroup-per-row form have ranged from
   ~2x to over 150x depending on row width, and layernorm-family kernels saw
   several-fold wins moving the same way. *If a kernel loops over
   `d_model`/`H*W`/`T` inside one invocation, it is leaving throughput on the
   floor.* Diagnostic signature when you are unsure: a per-row kernel whose
   achieved GB/s **rises with row count and falls with row width** is serial
   per row — its only parallelism is the rows themselves.

3. **A kernel that discards every invocation but one.** A grad-norm kernel
   opened with `if (gidx != 0u) { return; }` and was dispatched with
   `threads = 1`, so each parameter tensor's sum-of-squares was computed as
   dependent scalar loads on a single lane — a tiny fraction of a percent of
   achievable bandwidth, and the dominant share of that model's training GPU
   time. Fixed by a cooperative-tree kernel (one partial per workgroup) plus a
   small second pass over every tensor's partials: a four-order-of-magnitude
   win in situ, a large fraction on the whole training step, and the tree is
   also far more *accurate* than the serial accumulator. *Reductions want a
   cooperative tree plus a second small pass.* Fusing many small per-tensor
   dispatches on top of that was measured and rejected on one workload — once
   each is parallel, the dispatches are cheap enough that fusing them further
   buys nothing worth the complexity.

4. **A workgroup tile pays only where the amplification is large.** Adding a
   shared-memory tile to a coalescing-bound kernel made it *slower*: its
   uncoalesced side is amplified only modestly, and below some multiplier the
   occupancy cost of the shared memory exceeds the coalescing win. The same
   tile on a kernel with much higher amplification gave a solid win.
   *Estimate the amplification before reaching for shared memory.*

5. **A `var<workgroup>` array sized for the WORST case is an occupancy bug at
   every other case.** Workgroup memory is allocated statically per workgroup,
   so an accumulator array sized for the largest supported parameter costs that
   much whatever the caller actually asked for. A decode GEMV declaring
   `array<f32, 2048>` (its `m <= 32` worst case) reserved **8 KB per workgroup
   at every `m`**, capping residency at 12 of a possible 32 workgroups per SM -
   ~37.5% occupancy - and running at **36% of the card's measured memory roof**
   while it did. *Size a workgroup array by a `kernels::template` constant the
   selector sets from the shape, not by the contract's upper bound.*

   **Diagnostic, and it does not require trusting anyone's arithmetic about the
   hardware:** sweep the declared array size (halving) and time it. If shared
   memory is the limiter the curve *rises* at each halving and then goes
   **flat** - flat exactly where the per-SM block cap starts binding instead.
   Measured on a GP102: 8 KB → 4 KB → 2 KB rose (79.5 → 141.5 → 166.8 GB/s),
   then 1 KB and 512 B were unchanged. That shape is what proves the 96 KB/SM
   and 32-blocks/SM numbers on the actual device.

6. **A shared-memory read-modify-write in an inner loop is a dependency chain,
   not just extra stores.** `partial[i] = partial[i] + a*b` per `(k, m)` makes
   every accumulator wait a shared-memory round trip per k-step. Moving the same
   accumulators into registers on the same kernel was worth **2.7x** where
   shrinking the array alone was worth 2.1x - and the register version at
   *37.5%* occupancy beat the shrunk shared version at *100%*, i.e. the
   occupancy fix mattered mainly because it bought latency-hiding for a chain
   that should not have existed. *If a kernel accumulates into workgroup memory
   inside its hot loop, the array should hold only what must cross threads (the
   final fold), never the running sums.*

   **Registers need a COMPILE-TIME trip count, and the runtime-parameter dodge
   is measurably worse.** Named scalar accumulators guarded by uniform
   `if (p.m > 1u)` branches - runtime `m`, no array, no template - measured
   *slower than the plain shrink*, at every shape tried. Uniform,
   perfectly-predicted branches in the innermost loop still cost more than they
   save. So the register form is a `kernels::template` knob plus a **bucket
   ladder**, and the ladder is a measurement (§F.6): a single worst-case
   specialisation is a REGRESSION at small parameters - one compiled for 32 rows
   ran **0.44x** (2.3x slower) than the kernel it replaced at 1 row, because its
   cost tracks the compile-time bound and not the caller's actual shape.
   A power-of-two ladder won by ≥1.7x at every row count instead.

## D. Constraints that will bite you

- **No atomics, no subgroups, no f16** — a tree reduction is
  shared-memory + a second pass, never an atomic accumulate.
- **The CPU JIT rejects a FUNCTION-scope array inside a work-group kernel**
  (`array local in a work-group kernel is unsupported` - its per-invocation
  locals are SSA scalars). This is the *second* structural reason a kernel
  cannot be `@cpu yes`, alongside the barrier count below, and it is what makes
  a register-accumulator kernel a GPU-only SIBLING rather than a template
  variant of a portable one (`matmul_gemv` / `matmul_gemv_reg`). Both reasons
  are derived from the code by `scripts/build/kernelmeta.py::cpu`, so the
  declared `@cpu` cell cannot drift from what the JIT will actually do - and
  `wgsl_cpu::Jit` *skips* only these two, failing hard on anything else, so a
  real port bug can never hide as a skipped kernel. Re-verify such a claim
  before building on it; this one was still true, but the header asserting it
  had never been rechecked.
- **The CPU backend's Cranelift JIT supports ONE top-level barrier per kernel.**
  A textbook two-pass mean/variance will not JIT. The LayerNorm kernels use the
  *shifted* one-pass form (`K = x[row,0]`; `mean = K + S1/d`,
  `var = S2/d − (S1/d)²`) which avoids `E[x²]−E[x]²` cancellation and fits one
  barrier. A kernel that falls out of the JIT prints
  `wgsl-cpu: kernel "…" not JIT-compiled` — grep your test output for it.
- **A GEMM chunks only along the axis that is a contiguous ROW RANGE of its
  operands.** `step_sliced` binds sub-ranges, never strides, so the chunkable
  axis is whichever one indexes the *rows* of both the operand and the output.
  For `out = col . Wᵀ` (`matmul_reg3`) that is the position axis - which is
  exactly why the conv lowerings put positions first, and it is not a
  stylistic choice. The TN form (`out[n,k] = Σ_m a[m,n]·b[m,k]`, which is what
  a *transposed* conv wants because its contraction is the leading axis of
  both native operands) has NO chunkable axis at all: `n` slices `a`'s columns
  and `k` slices `b`'s columns, both strided. Such a lowering has to be
  *bounded* and capability-gated on the binding limit instead, or pay for a
  transpose.
- **`matmul_dw_reg` ACCUMULATES; `matmul_dw_reg_splitk` with `s = 1` is the
  same GEMM and ASSIGNS.** If a lowering writes its intermediate into a
  scratch buffer reused across dispatches, the accumulating one folds in the
  previous user's bytes. Zeroing the scratch instead costs a full extra pass
  over the biggest buffer in the pipeline; picking the `_splitk` sibling costs
  nothing.
- **`max_storage_buffer_binding_size` is a real, queryable limit** — on some
  in-support hardware it can be well under 2 GiB, not the full device memory.
  A whole-image im2col operand for a large convolution can exceed it easily —
  unbindable. Conv-as-GEMM therefore needs the **transposed** orientation
  (`y[HW,Cout] = col·Wᵀ`) so a spatial chunk is a contiguous row range of both
  operands. Never assume a buffer of a given size will bind; query
  `DeviceCaps` and chunk accordingly.
- **≤8 storage buffers, fp32 only, `@workgroup_size(64)`** (256 is the
  documented exception for register-tiled matmuls; justify anything else in a
  comment and honour `DeviceCaps::{max_workgroup_size, workgroup_reductions}` —
  queried, never assumed).
- **Non-ReBAR cards accrue wgpu staging per `write` until a blocking
  readback.** Multi-GB uploads must flush periodically (~1 GiB) or the device
  OOMs at well under its VRAM — `paramstore`'s upload loop does this.
- After adding/removing a `.wgsl`: **`make kernels-regen`**.

## E. Optimizing: measure first, because the guess is wrong

> **§F is the procedure**; this section is the evidence for why it starts with
> a profile. If you are here to make something faster, read §F first and come
> back for the killed hypotheses.

Across several rounds of optimization on this engine, **confident hypotheses
have repeatedly been wrong and the profile has repeatedly been right.** A
sample, with the number that killed each one:

| hypothesis | reality |
|---|---|
| per-dispatch overhead dominates | measured at a small fraction of a percent in one pass, a couple of percent in a dispatch-heavy backward — a low ceiling on what fusion could return either way |
| deeper K-blocking will fix the GEMM | a blocking factor **cancels** in the arithmetic-intensity formula — algebra error; the GEMMs were already well into the tens of percent of peak |
| the GEMMs are the bottleneck | attention was the dominant share in one model; GEMMs a small fraction |
| the text encoder is the unprofiled half worth attacking | it was a small slice of the whole pipeline; a different stage dominated |
| a bias/coalescing bug explains a flat-throughput kernel | flat throughput across every shape tested turned out to be its structural byte/FLOP ceiling, not a fault |
| a batch of small grad-norm dispatches needs fusing over an offset table | once each dispatch is internally parallel, the whole group is a couple of percent of a training step and fusing buys well under half a percent more |
| "small but free" kernel registrations are free just because the kernel itself is faster | **KILLED** on one such change: a clear per-kernel win, ZERO on the whole pass, because the affected dispatches could not move a pass dominated elsewhere. It also changed the output slightly (a reassociated sum) and added a barrier kernel to a model that must also run on `backend-cpu`, for no wall-clock benefit — reverted. **Before optimizing a kernel, check what fraction of the PASS it can possibly return** |
| composing several coalesced stages beats a fused kernel | **KILLED** on one such comparison — fusing won at every shape tested, because the composed form paid the same sector-amplification cost twice (once per permute) that fusing pays once. The margin grew with the operand's spatial extent, exactly as the underlying bandwidth model predicted |
| a flash kernel is at a third of its roof because `head_dim` is half the 128-wide compile-time tile, so half of every tile is zero-fill | **KILLED in one grep at the config.** The model dispatches `head_dim = 128` - exactly the tile width, no padding at all. A `kernels::template` specialisation would have compiled to the identical kernel. *A recorded root cause is a hypothesis with a citation, not a measurement; re-derive it from the config the model actually dispatches before building a fix on it* |

### E.0 Bracket every timed region with `poll_wait()` — or you are timing the host

A backend `submit` call with an empty clear list can simply append to a
pending queue without encoding or queuing anything; a timing loop of bare
`submit`s then measures host-side bind-group construction and reports it as
device bandwidth. This has produced throughput numbers *above the card's
physical bandwidth roof* — the tell that host time, not device time, was
measured. Any timed region must be bracketed by `Gpu::poll_wait()`, which
flushes the pending pass and blocks until the device is done.

If a measurement comes out faster than the device can move memory, you
measured the CPU. Compute the roof first and sanity-check against it before
believing a result.

So:

1. **Profile per kernel-kind before touching anything, and publish the table.**
   Harnesses to copy: a per-model `*_bench` binary that replays a dispatch
   sequence over shape-correct scratch, no weight load, printing per-kind ms,
   GFLOP/s, and % of peak; and a standalone kernel micro-bench (min-of-N).
   Per-group drains cost a small fraction of the total — cheap enough to trust.
2. **Compute the roofline for the shape**, then compare: achieved GFLOP/s vs
   peak, or achieved GB/s vs the measured bandwidth roof (`gpu_core::roof`,
   queried per device — never a hardcoded constant). A kernel at <5% of both
   is a bug; a kernel at its byte/FLOP ceiling is structural and needs an
   algorithmic change, not tuning.
3. **Re-measure the phase split after each fix.** The bottleneck moves: in one
   workstream a diffusion model's forward went from attention-dominated to
   GEMM-dominated, then a different phase (the VAE) became dominant, then the
   optimizer. Optimizing yesterday's profile is how you get a large win worth
   a fraction of a percent of the step.
4. **Correctness gates every time**: the model's parity test (cosine must not
   move), `make gradcheck`, and — for anything touching the optimizer or a
   backward — **trajectory equivalence**: identical loss curves before/after, on
   the same seed. A drifted reduction changes training outcomes without failing
   a test. Know which kind of reduction you have: `max`/`min` are associative
   **and exact**, so a cooperative rewrite is bit-identical and needs no
   tolerance at all; `+` reassociates, so a cooperative rewrite moves the last
   bits — usually for the better (closer to an f64 oracle) but never for free.
5. **Before deleting a "workaround for X" because X is now fixed, find what else
   the code depends on.** A host-side grad-norm path in one multi-GPU model was
   documented as dodging a slow device reduction; once that reduction was fixed,
   the host path looked like free cleanup. It was not: the clip there is over a
   cross-replica **sum** that exists only in host RAM, and `‖Σ g_r‖ ≠ f(‖g_r‖)`,
   so a per-rank device norm is a *different number* — a silent change to every
   data-parallel run. The comment was stale; the code was right. Write the
   comment so the next reader can tell the difference: **say what breaks if X
   goes away**, not just what X cost.

---

## F. The loop: how the big wins were actually found

§E says measure first. This section is the *procedure* — the order the steps go
in, and the decision at each one — reconstructed from runs that produced large,
multi-times wins on more than one model's forward and backward pass in a single
day. Follow it in order; most of the steps end the work early, and the ones
that end it early are the cheap ones.

### F.1 Profile per KERNEL KIND, and publish the table

Not per model, not per layer. Group contiguous runs of one kernel in submit
order so the sum of the parts is comparable to the whole, print both, and
`poll_wait()`-bracket every region (§E.0). A per-model `*_bench` binary that
covers a **backward** pass is worth having — many benches only ever cover a
forward.

Publish the table in the commit. Every number below started as a row in one.

**But the group table is an UPPER BOUND, not the cost.** Each group is drained
separately, so its number includes a queue round-trip and excludes the overlap
that kernel would have had in the real submit — on one backward pass the
grouped sum inflated the true whole-pass time by roughly 50%. Use the
table to RANK, and the whole-pass number to decide whether a fix worked. One
change looked like a big win in the table and moved the whole pass by nothing;
it was reverted (see `.agents/rules/lessons.md` #21).

**And profile at the width the model is actually RUN at.** A table taken at a
convenient smaller shape is not a conservative version of the real one - it is
a different ranking, and acting on it optimizes something nobody runs. On one
video DiT the token count per forward at a real generation resolution was
nearly 4x the count an earlier pass had profiled at; self-attention is O(T²)
where every other kernel in the block is O(T), so every share moved and the
top row changed identity. A pass aimed at the small table found the dominant
*host* stage, fixed it for a large local win, and returned a fraction of that
end to end. Frame count multiplies the NUMBER of forwards; RESOLUTION sets
tokens per forward, and tokens per forward is what decides kernel behaviour -
so the cheapest honest profile is the smallest frame count at the real
resolution, never the other way round.

A corollary for reading a cumulative profiler: when the harness prints running
totals rather than per-call ones, the number you want is the DIFFERENCE between
two tables, and which two is a decision. A warm/cache-hit arm and a cold/first
call are different shapes of work; quoting the wrong one, or a raw cumulative
total, silently folds one-off setup (head projections, connector routing) into
a per-layer figure.

### F.2 Ask what the top row is running at, against the roof

A percentage of the profile tells you where the time is. A percentage of *peak*
tells you whether it can be fixed. A GEMM shape running at a few percent of
peak is not a kernel that needs tuning — it is the wrong kernel.

And if the top row is already at its **memory** roof, no kernel change can
help. The only lever left is to move fewer bytes, and in the decode regime
(`m` in the single digits against an `[n, k]` weight matrix) the standard way
to do that is **batch the independent rows** - because a GEMV's cost is the
weight matrix, not the vector. Two independent sequences run as two `m = 1`
calls stream the whole matrix twice; run as one `m = 2` call they stream it
once, for the same arithmetic. So before reaching for a kernel, ask what else
in the caller is standing at the same position on the same weights.

Two things make this worth checking every time:

* **The answer is usually sitting right there.** A classifier-free-guidance
  pair, a speculative-decode candidate set, and any two concurrent serving
  sequences are all independent rows over one weight set. They look like two
  calls in the source because that is how the reference implementation wrote
  them, not because they have to be.
* **It is bit-identical, on both sides.** A batched GEMV gives every output
  its own accumulator and reduces `k` in the same order regardless of the row
  count, so `c[o, r]` does not depend on how many rows accompany it - on the
  device (`matmul_gemv`, one workgroup per output column) and on the host
  (`hostmath::linear_rows` → `backend_cpu::fast_ops::matmul_abt`) alike. That
  is a rare thing to be able to `assert_eq!` on, so gate it that way and get a
  work reduction with no tolerance attached. Do NOT assume it: `row_abt_avx2`
  register-blocks its column loop 4-wide, so it is worth a test that straddles
  that block (rows 1..8) rather than a reading of the source.

Where this lands matters as much as the fix (§F.7): batching belongs in the
shared row-batched primitive, not in one model's step function, so the next
caller inherits it.

### F.2b The dual of batching: how many of those rows are the SAME row?

§F.2 asks what else is standing on the same weights so `m` can go UP for free.
The mirror question, and it has been worth more: **of the `m` rows this call
already has, how many are distinct?**

A `[t, width]` table whose row `i` is a pure function of one scalar `key[i]`
has exactly `distinct(key)` different rows, however large `t` is. Compute
those, keep a `[t]` index, and scatter. It is not an approximation - equal
input bits produce the identical sequence of roundings, so the compact form is
BIT-IDENTICAL and gates with `assert_eq!` on bits, no tolerance.

Measured, on ltxv's per-token adaLN-single table (`[3520,4096] x
[36864,4096]ᵀ` on the host, plus a 519 MB upload of the result, once per
forward): the timesteps that key it are `denoise_mask * sigma`, so a plain
text-to-video step has **one** distinct row and an anchored or long-form one
has **two**. 10.26 s of host time per warm forward became 0.22 s, and the
upload became 147 KB.

Three rules this pattern comes with, each paid for:

* **Dedup GENERICALLY, do not special-case "they are all equal".** A uniform
  fast path plus a full-cost fallback gives up the entire win the moment a
  single token is conditioned - which is every image-anchored and every
  long-form generation. Deduplicating to the distinct set wins ~1750x in that
  case instead of 1x, and it removes the fallback branch (and its test debt)
  altogether: `distinct == t` degrades continuously to exactly the old cost.
* **Key on RAW BITS, not `==`.** `0.0 == -0.0` is true and they are different
  inputs; `NaN == NaN` is false and they are the same input. `f32::to_bits` is
  total and is the only relation that licenses "same key, therefore same
  answer".
* **The failure mode is the SCATTER, and a uniform test cannot see it.** With
  one distinct row every scatter is the same scatter. Mutation-verified here:
  rotating the gather by one token passes the uniform case and fails the
  two-value one. So the gate needs a distinct-count LADDER - 1, 2 interleaved,
  several reused, all distinct - and 2 has to be interleaved, or a scatter that
  assumed a contiguous split passes too.

Ask it wherever a per-token/per-position quantity is derived from something
coarser: timestep and noise-level embeddings, per-token conditioning tables,
per-sample class embeddings in a batch, RoPE tables over repeated positions.

### F.3 Before writing anything: is there already a faster sibling?

This is the highest-value question in the list and it costs one `grep`. It has
been the answer repeatedly across workstreams:

| what looked like new work | what it actually was |
|---|---|
| a model's attention is slow | a split variant of the same kernel existed, unregistered — several-times win |
| a cross-attention projection is slow | the tiled GEMM existed; a *selection rule* excluded it — over an order of magnitude on that shape |
| a model needs a faster GEMM | it already carried the register-tiled variant but dispatched the older one |
| the CPU GroupNorm fallback is slow | another crate had written a barrier-free two-stage reduction privately — several-times win |
| a video DiT's every RMSNorm is a top-five row | `rmsnorm_rows` existed, `model::block::rms_variant` already implemented the selection rule, and two other crates already registered it - that crate had simply never been wired up. ~10x on the row, for one registration and one call site |

Only when this comes back empty do you write WGSL.

**"A kernel family cannot express this shape" is a claim about the KERNELS,
not about the algorithm - recheck which it is.** One model's cross-attention
kept a materialized scores/softmax/apply trio behind a documented reason: the
flash family "computes bidirectional self-attention over one span of rows that
are simultaneously the queries, the keys and the values", so a different key
row set "is not expressible in this kernel family at all". Every word of that
was true of the kernels that existed - all of them derive both tile counts
from one `tcols`, and the one member with separate q/k/v buffers is also
causal. None of it was true of the online-softmax algorithm, which never cared
whether the two lengths were equal. The fused cross kernel turned out to be
the best bidirectional rung with two changes (three operand buffers instead of
one fused slab, two lengths instead of one) and nothing else changed at all -
same tiling, same register block, same barriers - for **7.6x** on that
model's whole cross-attention and a `[heads, nq, nk]` slab pair, 3.46 GB at
its real width, that stopped being allocated.

So when a note says a shape is inexpressible, check whether it names a
*structural* obstacle (a mask that must not exist, an operand layout no caller
can produce) or merely the current Params list. The second is a diff, not a
research problem.

### F.4 Profile the branch your hardware does NOT take

A capability-gated fallback is invisible on the machine that never takes it.
One GroupNorm implementation picked a cooperative reduction on GPU and a
**serial** one otherwise; the CPU backend reports `workgroup_reductions: false`,
so every conv-autoencoder in the tree ran the unmeasured branch there, several
times slower than the barrier-free path that already existed elsewhere.
Enumerate the branches and measure each.

### F.5 A/B for CORRECTNESS and speed in the same harness

A faster kernel that disagrees is not a faster kernel. Print `max|delta|`
beside the timings, always.

And compare against a **host oracle**, not just kernel-to-kernel: one GroupNorm
A/B first reported all three kernels disagreeing significantly, which turned
out to be the *harness* dispatching a `@workgroup_size(256)` kernel at 64
threads. It was caught only because a parity gate elsewhere was green, so the
harness had to be the liar. Kernel-vs-kernel agreement cannot tell you which
one is wrong.

### F.6 Sweep for the crossover; never guess a threshold

Where two kernels trade places, the threshold is a measurement. Sweep the
parameter that separates them and read it off:

    Cout      3     8     16     32     64    128
    direct 4.25  9.82  18.03  35.01  69.63 138.88     ms
    lowered 24.25 24.09 24.28  23.99  25.27  26.79
    ratio  0.18x 0.41x  0.74x  1.46x  2.76x  5.18x

Two corollaries, both paid for:

* **Each kernel pair gets its OWN threshold.** Reusing a forward-pass
  crossover for the backward left a real win on the table across a whole band
  of channel counts — the measured backward crossover was different from the
  forward's.
* **A selection rule is as much a bottleneck as a kernel.** A GEMM-selection
  guard sent dozens of dispatches per forward to the naive kernel on a
  threshold that was much too conservative; the real crossover was far lower.
  Profile the selection, not just the kernels.

And two more, from porting the 2D conv-as-GEMM lowering down to 1D:

* **A threshold is a property of the PAIR, not of the fast kernel.** The 1D
  lowering uses the same `matmul_reg3` as the 2D one, so its `Cout` threshold
  looked like a free carry-over - 32. Measured, it is **16**, because the two
  lowerings' *baselines* differ: 2D's direct side is `conv_bias_reg`, an
  `@opt 5` register-tiled conv at ~700 GFLOP/s, and 1D's is `conv1d`, a naive
  kernel at 2.2% of roof. A weaker baseline crosses over earlier. Carrying the
  number over would have left 1.9x-6.0x unclaimed across `16 <= Cout < 32`.
  The transposed pair diverged further still (it wins at *every* width
  measured, down to `Cout = 4`) and needed its own constant.
* **A sweep cannot see below its own threshold, so it will "confirm" whatever
  you wrote.** Once the selector is wired, both columns of the A/B return the
  *same* path below the threshold and the ratio reads a meaningless 1.00x -
  which looks like evidence and is not. Ship the force switch with the
  threshold (`BRAIN_CONV1D_GEMM=force`, the same shape as `BRAIN_NO_COOP_LN`)
  and sweep through it.

### F.7 Put the fix in the SELECTOR, so the next model inherits it

A fix in a shared kernel-selection function — a `pick_gemm`, a `gemm_variant`,
a capability branch inside a shared block module — reaches every model that
already calls it, including ones written later. A fix in one model's `model.rs`
reaches exactly that model.

### F.8 Gate it, then mutation-verify the gate

Write the correctness gate before wiring the fast path in (a `gradcheck` test
that compares the lowered form against the direct kernel across several
shapes, strided/padded/unpadded/asymmetric, on **both** backends). Then break
the thing on purpose and confirm the gate fails. A gate nobody has seen fail
is a hypothesis.

### F.9 Re-profile — the bottleneck moves, and that is the point

Every fix promotes the next row. One conv-gradient kernel went from the
dominant share of a backward pass to a negligible one, and the kernel that had
been invisible behind it became the new dominant share and the next target.
Stop when the top row is near the roof (§F.2), not when you are tired.

### The shape of the result

One documented run: a conv input-gradient kernel went from over a thousand
milliseconds across many calls to a couple of milliseconds in one call; the
backward pass roughly halved; the full training step dropped by a third; and
the backward/forward ratio fell from over 6x to under 4x.

Nothing in that came from a clever kernel. It came from measuring the right
thing, finding what already existed, and putting the choice where every model
sees it.

---

## The meta-rule

Several of the biggest wins in this engine's history were **one
already-known bug class, re-appearing in a model written later**. Where each
kind of finding belongs, so it stays found instead of being re-discovered:

- **The generalizable finding/lesson itself** (a bug class, a kernel-selection
  fix, an insight that applies beyond one model) belongs in this file (or
  `.agents/rules/lessons.md` if it's cross-cutting beyond kernels) the moment
  it's confirmed - that is what the next author actually reads before writing
  a kernel.
- **The session log with the real measured numbers behind it** (before/after
  timings, per-kernel profiles, what was tried and killed) belongs in that
  model's own `.agents/roadmap/<model>.md`, never in `docs/`.
- **`docs/` gets zero measured numbers, ever** - `docs/performance/overview.md`
  is methodology only, exactly to keep a documented finding from silently
  going stale the moment the hardware, driver, or kernel path it was measured
  on changes. This is also why a specific case study does not belong there
  even as an illustration: it accumulates the same way this rule's own
  `docs/performance/overview.md` case-study section once did, and had to be
  cut back out.

Where possible, the fix itself belongs in the *selector* so the next model
inherits it without knowing it exists.
