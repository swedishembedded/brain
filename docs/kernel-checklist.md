# Kernel checklist — read before writing, dispatching, or optimizing one

Every rule here was paid for with a real defect in this repo. The measured
number is attached so you can weigh it; the case study behind each is in
`docs/performance/overview.md`. This page is the pre-flight — that page is the
"why".

**Trigger**: you are about to add a `.wgsl`, dispatch an existing kernel from a
new model, or make something faster. Read the matching section first.

---

## A. Before you WRITE a kernel: does a good one already exist?

**The most expensive mistake in this repo is not a slow kernel — it is a fast
kernel nobody knew about.**

`gn_stats` was diagnosed and fixed for DIAMOND in 2025. `crates/vae` was written
afterwards, against the same kernel *name*, and silently inherited the slow one:
**2262 ms of a 6.5 s VAE decode, 159× when finally fixed**. Five instances of
one already-solved pattern were found in that single model.

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
   moves) and rewrites the *dispatch* in `step*`. `max_abs_rows` reached
   `qwen`, `zimage` and a `crates/flux2` owned by another agent with **zero**
   changes in any of them — 43.5 ms → 4.6 ms in the FLUX.2 int8 TE prefill. The
   bar is high on purpose — read that module's header before adding a row; a
   *sum* reduction fails the bit-identical rule and stays in the explicit seams,
   where a trajectory gate is visible at the call site. **Keep the recorded
   `StepMeta` in the caller's index space**: profilers map `meta.kernel` through
   their own kernel list, and the first version of this seam panicked
   `flux2_bench` by handing it an appended slot.

## B. Before you DISPATCH an existing kernel: read its contract

Two of this session's three real bugs were contract mismatches on kernels that
were already correct:

| defect | cost |
|---|---|
| `silu_mul` takes a **single `total`** param; passing `[rows, cols]` computed 1/9216th of the MLP | forward cosine **0.504** — silently wrong, not a crash |
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
   `flash_attn_bidir` held `q[128]` + `o[128]` per thread, indexed by a
   *runtime* `head_dim`: too big for the register file, and the runtime bound
   blocks unrolling, so both spilled to global-backed local memory — ~6 bytes
   per FLOP, a 58 GFLOP/s roof. It was **81% of the entire DiT forward**;
   **29×** when fixed with lane-splitting and a compile-time trip count.
   *Sizes that index thread-private arrays must be compile-time constants
   (`kernels::template` specialises them).*

2. **One thread per row is a coalescing bug.** A single invocation walking a
   whole row serialises the loads and wastes most of every memory sector.
   Measured fixes to the cooperative workgroup-per-row form: `gn_stats`
   **159×**, QK-norm via `rmsnorm_rows` **19.4×**, `layernorm` family
   **2.8–9.7×** (10.3× in situ for `gpt` decode), the int8 activation quant's
   `max_abs_row` → `max_abs_rows` **2.1–13.5×**. *If a kernel loops over
   `d_model`/`H*W`/`T` inside one invocation, it is leaving 3–150× on the
   floor.* Diagnostic signature when you are unsure: a per-row kernel whose
   achieved GB/s **rises with row count and falls with row width** is serial
   per row — its only parallelism is the rows themselves.

3. **A kernel that discards every invocation but one.** `gradnorm_sq` opened with
   `if (gidx != 0u) { return; }` and was dispatched with `threads = 1`, so each
   parameter tensor's sum-of-squares was `numel` *dependent scalar loads on one
   lane* — **0.08 GB/s, 0.023% of the P40's 346 GB/s**, and **87.2% of GPT's
   training GPU time**. Fixed by `gradnorm_part` (cooperative tree, one partial
   per workgroup) + `clip_coef_wg` (one small second pass over every tensor's
   partials): **2 122× in situ**, 8.2× on the whole training step, and the tree
   is also 4 orders of magnitude more *accurate* than the serial accumulator.
   *Reductions want a cooperative tree plus a second small pass. Fusing the
   per-tensor dispatches on top of that was measured and rejected — 77 launches
   are <0.25% of a step once each one is parallel; see
   `docs/performance/overview.md`.*

4. **A workgroup tile pays only where the amplification is large.** Adding a
   shared-memory tile to `im2col_at` made it **slower (273 → 311 ms)**: its
   uncoalesced side is amplified ~2.7×, and below roughly **3×** the occupancy
   cost of the shared memory exceeds the coalescing win. The same tile on
   `nlc_bias_nchw` (8× amplification) gave **4.4×**. *Estimate the
   amplification before reaching for shared memory.*

## D. Constraints that will bite you

- **No atomics, no subgroups, no f16** — a tree reduction is
  shared-memory + a second pass, never an atomic accumulate.
- **The CPU backend's Cranelift JIT supports ONE top-level barrier per kernel.**
  A textbook two-pass mean/variance will not JIT. The LayerNorm kernels use the
  *shifted* one-pass form (`K = x[row,0]`; `mean = K + S1/d`,
  `var = S2/d − (S1/d)²`) which avoids `E[x²]−E[x]²` cancellation and fits one
  barrier. A kernel that falls out of the JIT prints
  `wgsl-cpu: kernel "…" not JIT-compiled` — grep your test output for it.
- **`max_storage_buffer_binding_size` is 2047 MiB on the P40.** A whole-image
  im2col operand for a 512² Cin=256 conv is 2.4 GB — unbindable. Conv-as-GEMM
  therefore needs the **transposed** orientation (`y[HW,Cout] = col·Wᵀ`) so a
  spatial chunk is a contiguous row range of both operands.
- **≤8 storage buffers, fp32 only, `@workgroup_size(64)`** (256 is the
  documented exception for register-tiled matmuls; justify anything else in a
  comment and honour `DeviceCaps::{max_workgroup_size, workgroup_reductions}` —
  queried, never assumed).
- **Non-ReBAR cards (P40) accrue wgpu staging per `write` until a blocking
  readback.** Multi-GB uploads must flush periodically (~1 GiB) or the device
  OOMs at well under its VRAM — `paramstore`'s upload loop does this.
- After adding/removing a `.wgsl`: **`make kernels-regen`**.

## E. Optimizing: measure first, because the guess is wrong

In three rounds of optimization on this engine, **every confident hypothesis
was wrong and the profile was right.** Killed, with the number that killed it:

| hypothesis | reality |
|---|---|
| per-dispatch overhead dominates (~476 dispatches) | **0.03%** (0.0065 ms × 499) |
| deeper K-blocking will fix the GEMM | **BK cancels** in the AI formula — algebra error; GEMMs were already at 35.8% of peak |
| the GEMMs are the bottleneck | attention was **81%**; GEMMs 16% |
| the text encoder is the unprofiled half worth attacking | it was **1.23 s of 7.3 s**; the VAE was 88% |
| `conv_bias_reg` has a coalescing bug | flat ~700 GFLOP/s across **all 15 shapes** = its structural 0.75 byte/FLOP ceiling, not a fault |
| the optimizer's 385 grad-norm dispatches need fusing over an offset table | 385 was 77 tensors × **5 steps**; once each is a cooperative tree the whole grad-norm is **2.84 ms of a 840 ms step** and fusing buys <0.25% for a ParamStore relayout |
| `clip_coef` on one thread is fine, it sums a handful of numbers | true at 77 inputs (0.047 ms), **false at 11 586** (0.475 ms) — the cooperative reduction's own partials made a second cooperative kernel mandatory |

So:

1. **Profile per kernel-kind before touching anything, and publish the table.**
   Harnesses to copy: `crates/flux2/src/bin/flux2_bench.rs` (replays a dispatch
   sequence over shape-correct scratch, no weight load; per-kind ms, GFLOP/s,
   % of peak) and `crates/gpu-core/tests/bench_layernorm.rs` (standalone kernel
   micro-bench, min-of-N). Per-group drains cost ~0.2% — cheap enough to trust.
2. **Compute the roofline for the shape**, then compare: achieved GFLOP/s vs
   peak, or achieved GB/s vs 346 GB/s (P40). A kernel at <5% of both is a bug;
   a kernel at its byte/FLOP ceiling is structural and needs an algorithmic
   change, not tuning.
3. **Re-measure the phase split after each fix.** The bottleneck moves: the DiT
   went 81% attention → 80% GEMM, then the VAE became the dominant phase, then
   the optimizer. Optimizing yesterday's profile is how you get a 5.5× win worth
   0.6% of the step.
4. **Correctness gates every time**: the model's parity test (cosine must not
   move), `make gradcheck`, and — for anything touching the optimizer or a
   backward — **trajectory equivalence**: identical loss curves before/after, on
   the same seed. A drifted reduction changes training outcomes without failing
   a test. Know which kind of reduction you have: `max`/`min` are associative
   **and exact**, so a cooperative rewrite is bit-identical and needs no
   tolerance at all (`max_abs_rows` gates on `assert_eq!`); `+` reassociates, so
   a cooperative rewrite moves the last bits — usually for the better
   (`gradnorm_part` is 4 orders of magnitude closer to an f64 oracle) but never
   for free.
5. **Before deleting a "workaround for X" because X is now fixed, find what else
   the code depends on.** `model::parallel`'s host grad-norm was documented as
   dodging the serial `gradnorm_sq`, so once that was 2122× faster the host path
   looked like free cleanup. It is not: the clip is over the cross-replica
   **sum**, which exists only in host RAM, and `‖Σ g_r‖ ≠ f(‖g_r‖)`, so a
   per-rank device norm is a *different number* — a silent change to every
   data-parallel run. The comment was stale; the code was right. Write the
   comment so the next reader can tell the difference: **say what breaks if X
   goes away**, not just what X cost.

---

## The meta-rule

Three of the four biggest wins here were **one already-known bug class,
re-appearing in a model written later**. Cross-model findings therefore belong
in `docs/performance/overview.md` under a `Cross-model finding:` heading the
moment they are measured — and, where possible, the fix belongs in the
*selector* so the next model inherits it without knowing it exists.
