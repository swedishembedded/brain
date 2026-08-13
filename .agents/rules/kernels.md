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
   seam crashed a bench tool by handing it an appended slot.

## B. Before you DISPATCH an existing kernel: read its contract

Contract mismatches on kernels that were already correct have caused real bugs
more than once:

| defect | cost |
|---|---|
| `silu_mul` takes a **single `total`** param; passing `[rows, cols]` computed a small fraction of the MLP | forward cosine **0.504** — silently wrong, not a crash |
| `step_sliced` offsets/lengths are **f32 elements, not bytes** | SIGSEGV |

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

## D. Constraints that will bite you

- **No atomics, no subgroups, no f16** — a tree reduction is
  shared-memory + a second pass, never an atomic accumulate.
- **The CPU backend's Cranelift JIT supports ONE top-level barrier per kernel.**
  A textbook two-pass mean/variance will not JIT. The LayerNorm kernels use the
  *shifted* one-pass form (`K = x[row,0]`; `mean = K + S1/d`,
  `var = S2/d − (S1/d)²`) which avoids `E[x²]−E[x]²` cancellation and fits one
  barrier. A kernel that falls out of the JIT prints
  `wgsl-cpu: kernel "…" not JIT-compiled` — grep your test output for it.
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

### F.2 Ask what the top row is running at, against the roof

A percentage of the profile tells you where the time is. A percentage of *peak*
tells you whether it can be fixed. A GEMM shape running at a few percent of
peak is not a kernel that needs tuning — it is the wrong kernel.

### F.3 Before writing anything: is there already a faster sibling?

This is the highest-value question in the list and it costs one `grep`. It has
been the answer repeatedly across workstreams:

| what looked like new work | what it actually was |
|---|---|
| a model's attention is slow | a split variant of the same kernel existed, unregistered — several-times win |
| a cross-attention projection is slow | the tiled GEMM existed; a *selection rule* excluded it — over an order of magnitude on that shape |
| a model needs a faster GEMM | it already carried the register-tiled variant but dispatched the older one |
| the CPU GroupNorm fallback is slow | another crate had written a barrier-free two-stage reduction privately — several-times win |

Only when this comes back empty do you write WGSL.

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
