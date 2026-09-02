# kernel-performance - roadmap

The cross-cutting kernel/execution-architecture campaign: closing the gap
between "a substantially-better-than-naive WGSL inference engine" (which brain
already is - register-tiled FP32 GEMMs, DP4A INT8 at 5/5, paged KV, INT8 KV,
prefix caching, continuous batching, device-side greedy decode) and an engine
architecturally capable of approaching peak throughput on the hardware it runs
on. Triggered by a 2026-09 source-level audit plus three independent
explorations of the `latest` branch (kernel catalogue + selector, the Qwen
serving hot path, and backend/collectives/optimizer/CPU/dtype), all of which
re-derived their findings from the tree rather than from prior documentation.

Scope split from **[`completion-plan.md`](completion-plan.md)`#Phase 5`**,
which stays *model-specific* (ranked by measured cost against one model's own
baseline). This ledger is the engine work underneath all of that: it is not
about any one model, and per-model detail that this campaign produces still
lands in that model's own `.agents/roadmap/<model>.md`.

---

## How this ledger was built, and what it is measured against

Every finding below was re-derived from the tree on 2026-09-01 (`git log -1`:
`a438b766`), not carried over from `AGENTS.md` prose - §3 of the working rules
this campaign inherits is explicit that ancient ledger claims are not trusted
without re-measurement. Where a claim contradicts something `AGENTS.md`
asserted, `AGENTS.md` is corrected in the same change that proves it, per that
file's own "write down what you learned" rule.

**Hardware this ledger's own measurements come from**: 2x Tesla P40 (SM 6.1
Pascal - DP4A yes, **no tensor cores, no async copy, fp16 at 1/64 rate, no
bf16**) and a Xeon E5-2690 v3 (Haswell - **AVX2/FMA only, no AVX-512, no VNNI,
no AMX**). Phase 8 and the matrix-engine half of Phase 1's architecture
descriptor target hardware this box does not have; see "The hardware-harness
contract" below for how that is handled without either skipping the work or
faking a measurement.

## Verified findings this campaign is built on

| Finding | Evidence |
|---|---|
| **194 of 433 kernels (44.8%) are rated `@opt` 1 or 2** - 53 at 1/5, 141 at 2/5 | `docs/reference/kernels.md` |
| Paged attention materialises a full `[batch, n_heads, cap]` f32 scores **and** probs slab, three dispatches, sized by context *capacity* not live seqlen | `crates/qwen3/src/serve.rs:359-360,543-544,1213-1227` |
| `decode_softmax_batched` launches `b*nh` threads total and walks each row **serially three times** | `crates/kernels/wgsl/decode_softmax_batched.wgsl` |
| A 4/5 fused `flash_attn_causal_gqa` exists but **`qwen3::serve` never dispatches any flash kernel** - its only production caller is `qwen3omnimoe::thinker` | `crates/qwen3/src/serve.rs:177-202` (`gqa_scores: block::UNREGISTERED`) |
| `run_batched` ends with `gpu.read(&xn_final, bsz*d_model)`; chunked prefill then discards all but the last row | `crates/qwen3/src/serve.rs:892-895, 1555-1558` |
| Admission LM head is a **host** `matvec_par` over a full fp32 `[vocab,d]` host copy - "the one remaining host head" (its own comment) | `crates/qwen3/src/serve.rs:460-462, 1638-1649` |
| `qwen35::serve` has **no device head at all** (admission *and* decode on host) - a regression against qwen3 | `crates/qwen35/src/serve.rs:291-296` |
| QKV is **three** GEMMs, gate/up is **two**; no fused weight exists in qwen3/qwen35 | `crates/qwen3/src/serve.rs:1189-1191, 1234-1235` |
| Serving tape is `Vec::new()` per step with a fresh uniform + bind group per dispatch; **no capture/replay anywhere on the serving path** | `crates/qwen3/src/serve.rs:1169, 1246-1249` |
| Vulkan emits a **blanket `MemoryBarrier` between every consecutive dispatch**, and every flush creates a fresh `VkFence`, blocks, destroys it, frees the command buffer | `crates/backend-vulkan/src/lib.rs:964-981, 1125-1150` |
| No timeline semaphores, no submissions in flight - `queue_lock`'s own doc: "every submit here is already synchronous submit+fence-wait, never pipelined" | `crates/vulkan/src/context.rs:172-190` |
| `Collective` takes and returns owned `Vec<f32>`; `HostCollective` reads each shard to host and reduces in a **scalar loop**. No NCCL/RCCL/device-resident/async path anywhere in the tree | `crates/model/src/collective.rs:33-117` |
| Optimizer is **`3P+1` dispatches** plus **`P` separate 9-word `gpu.write`s per step** (`P` = tensor count); on wgpu each write after the first costs an empty `queue.submit(None)` | `crates/optim/src/lib.rs:61-137, 192-205` |
| `select::Op` has **8 variants**; attention, paged attention, softmax, conv2d, embed and MoE are entirely outside the selection seam | `crates/backend-api/src/select.rs:33-59` |
| `AutoTuner` searches **at most 3 candidates** and picks an implementation *family* - no tile size, workgroup size or pipeline depth is searchable | `crates/backend-api/src/select.rs:590-649` |
| `kernels::template` **can** rewrite `@workgroup_size` and any `const`, but **no call site does** - the only numeric knob in production use is `MREG` on two GEMV kernels via a fixed bucket ladder | `crates/kernels/src/template.rs`; `crates/gpu-core/src/upgrade.rs:130-192` |
| `Ops::matmul` resolves through a **fixed internal** `CachedSelector<DefaultSelector>` with no injection point, so the measured tuner reaches only `qwen3::serve` | `crates/qwen3/tests/no_kernel_names.rs`'s own scope note |
| Q4 uses **zero `dot4I8Packed`** - 8 scalar MACs per weight word - and `matmul_q4_gemv` still has the `array<f32,2048>` shared-memory occupancy bug that `matmul_i8_gemv_reg` already fixed for int8 | `crates/kernels/wgsl/matmul_q4_{dyn,gemv}.wgsl` |
| MoE is `5 x n_experts` dispatches per layer (**1280/layer** at 256 experts); the compact path is host-scanned with **one submit per expert** (~6100/forward at GLM scale); no indirect dispatch exists anywhere in the engine | `crates/model/src/moe.rs:255-284, 964-976, 1065-1131` |
| Storage tiers only: BF16/F16 decode to f32 inline (f16 costs a ~10-op decode **per weight element**); FP8 is host-side checkpoint decode, not a `DType`; FP4/NF4 absent; native-f16 compute exists as an **unwired proof-of-concept** | `crates/model/src/ops.rs:378-407`; `crates/kernels/src/template.rs:705-766` |
| CPU has 9 AVX2 functions and **exactly one** AVX-512 function (itself untested on any box this repo has run on); no VNNI, BF16, AMX or NEON; `int8_dot: false`; the ISA if-ladder is re-evaluated **per row** | `crates/backend-cpu/src/fast_ops.rs`, `crates/backend-cpu/src/lib.rs:1013-1021` |
| **The profiler silently stops timing at 8192 dispatches** - so MoE and deep-model passes currently have *no* per-kernel attribution | `crates/backend-vulkan/src/lib.rs:956` |
| `crates/autodiff` is a 483-byte doc-comment-only file with zero consumers, still declared in the workspace | `crates/autodiff/src/lib.rs` |

## Decisions this ledger encodes

1. **Provider seam now, native packs later.** WGSL is the portable reference
   and correctness oracle, not the performance ceiling. Phase 8 builds the
   `OperatorProvider` ABI and the architecture descriptor; **no vendor pack
   ships in this campaign.** See the amended WGSL bullet in `AGENTS.md`'s
   conventions section.
2. **Build for hardware this box lacks, with a Zephyr-style harness contract.**
   FP8/FP4, native f16/bf16 compute, VNNI/AMX/AVX-512 and matrix-engine paths
   are implemented and capability-gated regardless of what this box can run.
   A test that cannot execute here **skips loudly** (see "The hardware-harness
   contract" below), is recorded in a machine-readable capability ledger, and
   states plainly that its behaviour is unvalidated on this box and may fail on
   the box that can actually run it. Current hardware is not the target this
   campaign is aimed at.
3. **Do not trust ancient ledger claims.** Every row in the findings table
   above was re-derived from the tree, not carried over. As this campaign
   corrects stale `AGENTS.md`/roadmap claims, the correction lands in the same
   change that proves it, named as such.
4. **The audit is the candidate set, not the sequence.** `AGENTS.md` §E already
   records that on this engine "every confident hypothesis has been wrong and
   the profile has been right" - including a killed hypothesis that
   per-dispatch overhead dominates, and M22 (`qwen35.md`) measured the qwen35
   decode path at ~2% host time, which bounds what graph capture can return
   there. Ordering *within* a phase is set by a fresh profile, not by the
   audit's prose ranking; a candidate the profile shows cannot move a real pass
   is recorded as **killed**, which is a successful outcome of this process,
   not a failure to build it.

## The hardware-harness contract (decision 2)

Modelled on how firmware test suites gate on a hardware harness rather than
skip silently: `brain_testutil::skip_unvalidated_capability(cap, reason)`
(built in M0.3) prints a warning naming the capability and the hardware it
needs, states plainly that the behaviour is unvalidated on this box and may
fail on hardware that has it, and appends a row to a machine-readable ledger
(`make test/capability-report` renders it). Nothing here may be silenced by a
flag that makes it fatal by default - it is missing *hardware*, not a bug - but
`BRAIN_REQUIRE_CAPABILITIES=<list>` exists for the box that does have the
capability, so that box's CI can promote the named skip to a hard failure the
same way `BRAIN_REQUIRE_FIXTURES=1` does for `brain_testutil::skip`.

---

## Phase structure

Full milestone detail (file paths, exact gates, commit counts) lives in the
approved implementation plan for this campaign; this ledger tracks status and
records measured deltas and killed hypotheses as phases close. Phases in
dependency order:

- **Phase 0 - Trustworthy measurement.** The profiler dispatch-count bug, a
  published baseline every later phase is measured against, the
  hardware-harness contract, and a small technical-debt sweep found along the
  way (autodiff deletion, two per-call-hoisted lookups, one host allocation in
  the MoE dispatch loop).
- **Phase 1 - Make the wrong kernel unreachable.** Widen `select::Op` to every
  dispatched family (attention, paged attention, softmax, conv2d, embed, MoE),
  give `Ops` an injectable selector and delete the bespoke
  `qwen3::serve`-only exception, add a workspace-wide gate against hand-picked
  kernels cross-referenced against the kernel catalogue, extend the zero-edit
  `gpu_core::upgrade` seam. This is the architectural goal that stops this
  campaign's own wins from being silently lost by the next model - the repo's
  most expensive recorded defect class (`gn_stats`, 159x).
- **Phase 2 - Fused paged attention.** `paged_flash_decode`/`_prefill`: online
  softmax over the paged block table, no materialised scores/probs, fp32 and
  int8 KV, wired through the Phase 1 selector, with the scratch shrunk once
  it's live.
- **Phase 3 - Remove host synchronisation from serving.** Submit-only prefill
  chunks, a device admission LM head, coalesced per-step host writes, and
  bringing `qwen35::serve` up to `qwen3::serve`'s already-fixed contract.
- **Phase 4 - Transformer-block fusion.** Fused QKV, fused gate/up, fused
  QK-norm+RoPE+KV-append, fused RMSNorm+quantisation - each required to report
  its measured memory-traffic delta and recorded as killed if it does not move
  a real pass, per decision 4.
- **Phase 5 - The `@opt` 1-2 kernel sweep.** 194 kernels grouped into families
  (norm fwd/bwd, attention backward, conv, MoE, Q4, MLA/DSA/GDN,
  reductions/losses/router), each run through `kernels.md` §F end to end.
- **Phase 6 - Runtime execution.** Vulkan per-buffer dependency tracking
  (replacing the blanket barrier), asynchronous submission (timeline
  semaphores, persistent command buffers, submissions in flight), graph
  capture/replay with shape buckets, a multi-tensor optimizer.
- **Phase 7 - Distributed.** Device-resident asynchronous collectives
  replacing the `Vec<f32>` host-staged API, communication/backward overlap.
- **Phase 8 - Precision tiers and the provider seam.** The `OperatorProvider`
  ABI and architecture descriptor, schedule-space autotuning, real low-precision
  compute tiers (native f16/bf16, FP8, FP4/NF4), CPU ISA packs (AVX-512, VNNI,
  AMX) - all harness-gated per decision 2.

---

## Done

### M0.0 - This ledger, and reconciling `AGENTS.md`

Created this file; cross-linked from `completion-plan.md`'s Phase 5 (which
stays model-specific) and from `AGENTS.md`'s task table. Amended `AGENTS.md`'s
"fp32 arithmetic only, core compute only" bullet to record WGSL as the portable
reference/correctness-oracle rather than a claim that no operator may ever have
a second implementation, and to name the `OperatorProvider` seam (Phase 8) as
the sanctioned extension point once it exists - until then, that section's
constraints hold everywhere with no exceptions, which the bullet says
explicitly so a partial Phase 8 landing can never be read as license to bypass
them early.

### M0.2 - The baselines this campaign is measured against

Checkpoint-free profiles via existing infrastructure (`qwen_bench serve` and
`vqgan_bench`) - no new profiler code needed, `qwen_bench serve`'s `[rows]`
parameter already produces a decode-shaped (`rows=1`) or prefill-shaped
(`rows=N`) served step through the real `qwen3::serve::Engine` tape at
Qwen3-0.6B's real shape, on random weights. Measured on this box (2x Tesla
P40, `BRAIN_DEVICE=gpu`, measured roofline 10297 GFLOP/s / 285.4 GB/s DRAM).
**Per this repo's own convention** (`scripts/gates/*-perf-baselines/` is
gitignored - `.gitignore:68`, `check-large-files.sh` rule 2 - a dev box's
absolute numbers are one machine's snapshot, not portable source), the raw
per-kernel tables are NOT committed; they live locally at
`scripts/gates/kernel-campaign-perf-baselines/*.txt` (reproducible via the
commands below) and the measured numbers are recorded here as prose,
matching `qwen35.md`'s own M22 precedent.

**Decode** (`qwen_bench serve 1 20 512`, 590 dispatches, 18.14 ms/step,
55 rows/s): the paged-attention triad this campaign's Phase 2 targets -
`decode_softmax_batched` (22.6%) + `paged_decode_apply_batched` (16.2%) +
`paged_decode_scores_wg` (2.8%) - is **41.6% of the whole decode pass**,
with the profiler's own defect flag firing on two of the three
(`decode_softmax_batched` at 0.2% of its memory roof against a 35% floor,
`paged_decode_apply_batched` at 7.2%). `matmul_gemv` is the single largest
line (43.3%) at 80.1% of roof - already well optimised, confirming the
attention triad, not the GEMV, is this shape's actual opportunity.

**Prefill** (`qwen_bench serve 128 20 512`, 786 dispatches, 132.18 ms/step,
968 rows/s): the same triad - `paged_decode_apply_batched` (29.4%) +
`paged_decode_scores_wg` (27.5%) + `decode_softmax_batched` (4.5%) - is
**61.4% of the whole pass**, ahead of the GEMM family entirely
(`matmul_reg3_splitk` + `dw_splitk_reduce` together 35.7%). This is real,
first-hand confirmation (not the audit's architectural inference) that
fused paged attention is this box's single highest-value target, at BOTH
the decode and prefill regimes.

**Training step** (`vqgan_bench 256 5`, 256x256, latent 8x8): forward
(409.21 ms) is 96.9% one kernel, `conv_bias_reg`, at only 3.5% of its
compute roof (DEFECT, floor 30%) - a register-tiled kernel that is
nonetheless far under its own roof at this shape, worth a dedicated look
before assuming register-tiled means roof-bound. Backward (456.66 ms) is
led by `col2im` (27.0%, 9.3% of memory roof) and `bias_grad` (19.1%, 1.3% of
memory roof) - both real `@opt` findings for Phase 5's conv-family sweep
(M5.3), not yet gated by name here since that phase profiles and selects
per kernel, not per model.

Reproduce: `make build/release`, then
`BRAIN_DEVICE=gpu ./target/release/qwen_bench serve 1 20 512`,
`BRAIN_DEVICE=gpu ./target/release/qwen_bench serve 128 20 512`,
`BRAIN_DEVICE=gpu ./target/release/vqgan_bench 256 5`.

### M0.3 - The hardware-harness contract

Added `brain_testutil::skip_unvalidated_capability(cap, reason)` beside the
existing `skip`/`skip_unavailable` in `crates/testutil/src/lib.rs`: prints a
prominent stderr warning naming the capability and why it's unvalidated here,
states plainly the result MAY FAIL on hardware that has it, and appends a
tab-separated row (`cap`, `reason`, `#[track_caller]` call site - no
wall-clock timestamp, so it stays deterministic-friendly) to a ledger at
`$BRAIN_CAPABILITY_LEDGER` (default `<repo>/out/capability-ledger.tsv`).
Non-fatal by default; `BRAIN_REQUIRE_CAPABILITIES=<comma-separated caps>`
promotes a named capability's skip to a hard failure, mirroring
`BRAIN_REQUIRE_FIXTURES` but keyed by a list since capabilities are graded
per-box rather than one binary present/absent fact. `make
test/capability-report` (`scripts/gates/capability-report.sh`) renders the
ledger as a table (capability, skip count, reasons, call sites). TDD: a red
test asserting non-panic + ledger row + panic-under-`BRAIN_REQUIRE_CAPABILITIES`
went green against the implementation; `crates/testutil`'s full suite and
`cargo clippy --all-targets` stay warning-free. Documented in
`.agents/rules/testing.md` (prose + the env-var tables). No caller in the tree
uses this yet - that lands with the Phase 8 work it's built ahead of (FP8/FP4,
native f16/bf16, VNNI/AMX/AVX-512), per decision 2.

### M0.1 - The profiler stops silently dropping timing above 8192 dispatches

`backend-vulkan::flush()` gated its timestamp query pool on `steps.len() <
MAX_TIMED_DISPATCHES` (8192) and skipped timing the whole batch above it with
no warning - exactly the MoE-scale batches (tens of thousands of dispatches
per forward) that most need per-kernel attribution went completely
unattributed. `MAX_TIMED_DISPATCHES` was never a queried Vulkan device limit,
so the fix is real chunking, not a warning: `flush` now splits an oversized
batch into `ceil(n / MAX_TIMED_DISPATCHES)` bounded sub-batches
(`flush_chunk`), each its own submit+fence-bounded timestamp bracket, folding
every sub-batch into the same per-kernel accumulator. The untimed path is
byte-for-byte unchanged (one chunk = the whole batch when timing is off).
New test `kernel_times_attributes_every_kind_above_the_query_pool_capacity`
(8300 mixed dispatches, RED against the pre-fix code, GREEN after) pins the
contract; the full `backend-vulkan`/`backend-cpu`/`gpu-core` suites and
`cargo clippy --all-targets` stay green. Lesson recorded as
`.agents/rules/lessons.md` #81. Commit `1e930207` (implementation + test).

### M0.4b - Hoisted `BRAIN_VK_SERIAL`/`BRAIN_VK_NO_SERIAL` out of `flush()`

Same file as M0.1: both env vars were read via `std::env::var` on every single
`flush()` call rather than once per process. Resolved each once via a
`OnceLock` (`vk_serial_forced`/`vk_serial_disabled`), matching
`backend_api::select`'s `BRAIN_NO_COOP_LN`/`BRAIN_NO_COOP_GRADNORM`
convention. Pure hoist - same resolution semantics and fallback order, no
behavior change, confirmed by the full `backend-vulkan` suite staying green.
Commit `1ab6f562`.

### M0.4a - Deleted the dead `crates/autodiff` placeholder

483-byte doc-comment-only file, zero consumers (verified by grep before
deletion), still declared in the workspace. Removed the crate, its workspace-
member entry, its `Cargo.lock` entry, and its `AGENTS.md` row (the
placeholder note now names only `crates/timeseries`). `make build/release`,
`make check/doc-links`, `make check/scripts` all green. Commit `6eee2197`.

### M0.4c - CPU ISA tier resolved once per call, not per row/chunk

`crates/backend-cpu`'s `avx512_available()`/`avx2_available()` if-ladder was
re-evaluated inside every per-row/per-chunk hot-loop closure across
`fast_ops.rs` (`silu`, `silu_mul`, `matmul_abt`, `affine_sigmoid_inplace`,
`affine_silu_inplace`, `bn_eval`, `axpy`, `scale_add`,
`moe_linear_gated_fwd`). `is_x86_feature_detected!` already caches its CPUID
probe internally, so this was never a correctness bug - just wasted
if-ladder/closure-capture work per iteration. Added `fast_conv::IsaTier` +
`isa_tier()` (a `OnceLock`, resolved once ahead of the loop) and moved every
call site onto it - a pure hoist, proven by the existing `*_matches_scalar`
bit-identity tests staying green (no path selection changed). Commit
`8d0f8115`.

### M0.4d - Precomputed MoE per-expert weight-name tables

`qwen35moe`'s `moe_sublayer{,_bwd}`/`moe_sublayer_decode_sparse` and
`deepseek2`'s `decode_at` (the token-by-token hot loop; `build_forward`/
`build_backward` fixed too, cheap and one-time) `format!`-allocated
`blocks.{l}.mlp.experts.{e}.{gate,up,down}.weight` per expert per layer per
forward pass. Both models now build a `[layer][expert] ->
(gate,up,down)` name table once at construction and index into it. No
change to `ParamStore`/weight-lookup architecture, dispatch order, or
numerics. Verified via `check_qwen35moe`/`check_qwen35moe_lora` gradcheck,
`deepseek2`'s 8 finite-difference tests, and both crates' full suites +
clippy, all green. Commit `ca4b6c00`.

### M1.1's embed finding - `gpt2`'s untiled `EMBED` dispatch, tiled

The bug M1.1's per-family verdict table named and scoped out as "a one-line
fix unrelated to a new `Op` variant": `crates/gpt2/src/model.rs` dispatched a
bare `EMBED` kernel against the whole `[vocab, d_model]` `tok.weight` table in
one untiled, uncapability-checked dispatch, in both the batched
`forward_steps` embed stage and the per-token `decode_at` incremental-decode
embed. Every other decoder-LM in this repo (`qwen3`, `lfm2`, `t5encoder`)
tiles this lookup via `model::block::vocab_tiles_on` + `EMBED_TILE`
specifically because a large enough vocab table exceeds
`max_storage_buffer_binding_size` and fails `create_bind_group` outright -
`qwen3::model`'s own `embed_tiled` doc names the exact failure. `gpt2` was
"safe" only because its vocab (`calculator`/`reverser`/`shakespeare_char`,
all well under 100 tokens) has never been large enough to hit that limit, not
because it was tiled.

Ported `qwen3::model`'s `embed_tiled` pattern into `gpt2::model::Gpt`: a new
`vocab_tiles`/`embed_tiled` helper pair binds `tok.weight` as vocab-tile
sub-ranges via `step_sliced` + the already-shared `EMBED_TILE` kernel
(registered in `gpt2`'s own `PIPELINES`), and both call sites (`forward_steps`
and `decode_at`) now go through it. `pos.weight`'s embed (position table, not
vocab-scale) is untouched. `vocab_tiles_on` degenerates to one `(0, vocab)`
tile at every vocab size this crate ships, so the change is a no-op at
current scale by construction - confirmed by an explicit before/after A/B (a
throwaway example dumping an FNV-1a hash of a batched forward's full logits
and of a 10-step incremental-decode's reconstructed logits over a fixed
seed/config, run once against the pre-change tree via `git stash` and once
against the post-change tree): both hashes matched bit-for-bit
(`939e84c7ba51b31f` logits, `40c911b3e32d11e6` decode). Full `brain-gpt2` test
suite green (22 real tests, including `kv_step_matches_full_recompute`,
`cpu_register_equals_cpu_naive`, `dp_grad_parity_gpt`,
`shard_forward_and_grad_parity_gpt`, and the `convergence` suite), zero
warnings on `cargo build`/`cargo clippy -p brain-gpt2 --all-targets`. Does not
touch `crates/backend-api/src/select.rs`.

### M1.1's paged-attention milestone - `Op::PagedAttention`, a killed `kv_int8` "fix", and a second recalibration

`Op::PagedAttention` lands with exactly the shape the table below predicted:
`WorkgroupPerOutput` (`paged_decode_scores_wg`) vs `Reference`
(`paged_decode_scores_batched`), capability-only, no row/col gate - `Op::
MaxAbsRow`'s shape verbatim. The one addition the table did not anticipate:
the INT8 KV tier (`paged_decode_scores_i8_batched`) is a THIRD physical
kernel with no int8-cooperative sibling at all, so it needed its own
`candidates()` arm rather than inheriting `Op::MaxAbsRow`'s - reusing that
arm verbatim would have let an I8-tagged shape fall through to
`Reference`'s F32 physical kernel (which cannot read a packed pool) the
moment `WorkgroupPerOutput` got filtered out for lacking `int8_dot`. The
first pass at that arm mapped I8/Q4 to `KernelVariant::PackedInt8` (which
`requires()` unconditionally demands `int8_dot` for) - wrong, caught by
re-reading the actual kernel source per `kernels.md` B rather than trusting
the arm's own first draft: `paged_decode_scores_i8_batched` dequantizes its
pool with plain scalar WGSL bit-unpacking, no `dot4I8Packed` call anywhere,
and its header says `@cpu yes, @gpu yes` - exactly as portable as the
float `Reference` kernel beside it, unlike `Op::MatMul`'s genuinely
DP4A-bound `matmul_i8*` family that `PackedInt8` actually models. Fixed to
`Dtype::I8 | Dtype::Q4 => vec![Reference]` (its own physical kernel, no
capability gate at all). TDD: the new test (renamed twice along the way to
`paged_attention_scores_is_cooperative_at_every_shape_and_i8_never_gates_
on_int8_dot`) was written first and failed to compile before the variant
existed. Commits `343a1019` (the variant, with the wrong I8 arm) and
`b5adece5` (the correction).

**The first bug the table named does not exist as stated - caught only by
checking the actual kernel source, not by trusting the claim.** The table
said `qwen3::serve`'s `kv_int8` branch "never checks `caps.numeric.int8_dot`
… unlike its own `weights_int8`/`w8_on` sibling, which does" and implied
this was a live correctness bug because the int8 KV kernels "need
`dot4I8Packed`" the way the packed GEMMs do. A first pass gated `kv_int8`
exactly that way (commit `7e76a29f`) and it was WRONG: none of
`paged_decode_scores_i8_batched`, `paged_decode_apply_i8_batched`, or
`paged_kv_append_i8_clipped_batched` call `dot4I8Packed` anywhere - all
three dequantize/pack with plain scalar WGSL bit manipulation and are
`@cpu yes, @gpu yes` in the catalogue, exactly as portable as the fp32 KV
path. `weights_int8`/`w8_on` gates because `matmul_i8_dyn`/`matmul_i8_gemv*`
genuinely do call `dot4I8Packed`; `kv_int8` has no such kernel anywhere in
its path, so there was never a capability precondition to check, and the
original ungated code was correct. Gating it anyway was a real regression:
on `backend-cpu` (`int8_dot: false`), a `kv_int8: true` request would
silently degrade to fp32 KV even though the int8 KV kernels work correctly
there - caught by 5 failing `BRAIN_DEVICE=cpu cargo test -p brain-qwen3
--lib serve::` tests (exactly the run `make parity` exercises). Reverted
in commit `3c690652`, restoring the original doc comment's claim ("int8 KV
has no capability gate to fall back from") which was correct all along,
with a note on why so the next reader does not re-derive the same false
assumption. Full `brain-qwen3` suite green on both `BRAIN_DEVICE=cpu` (35
`serve::` tests) and `BRAIN_DEVICE=gpu` (98 tests) after the revert. This
is exactly decision 3's discipline turned on the campaign's OWN claim
rather than an inherited one: re-derive from the tree, even when the
claim is this milestone's own prompt, not a stale doc.

`qwen3::serve` was also the ONLY real caller of this kernel family in the
tree (`Ops::decode_scores_batched`'s dtype-axis façade in `crates/model/src/
ops.rs` is documented as a separate, unwired bf16 tier - "NOT wired into
qwen3::serve::Engine by this phase"), and it already implemented `Op::
PagedAttention`'s exact rule by hand. `model::block::paged_scores_variant`
was added mirroring `rms_variant`/`ln_variant`/`softmax_variant`'s shape
(the thread-count formula differs from those three: the cooperative kernel
owns `PAGED_SCORES_PER_WORKGROUP` scores per workgroup, not one row, so the
helper takes `batch_heads`/`cap` rather than a row count), and `qwen3::
serve`'s hand-rolled check was migrated onto it - the same shape as `Op::
Softmax`'s wan/ltxv migration. Pure refactor, verified via three repeated
full `brain-qwen3` runs (98 passed each; an intermittent SIGSEGV-on-exit
seen once during iteration reproduced with BOTH the old and new dispatch
code and is a pre-existing GPU-driver-teardown flake per `gpu_core::
testgpu`'s own doc, not caused by this change) and `brain-model`'s full
suite (152 passed). Commit `eb36160a`.

**The second bug the table named does not exist as stated - recalibrated
the same way the embed/conv2d/MoE verdicts above already were.** The table
said "`qwen35`/`qwen35moe` never register or reach the `workgroup_
reductions`-gated cooperative scores kernel `qwen3::serve` gets." Re-derived
from the tree: both crates register `paged_decode_scores_batched` in
`model.rs`, but ONLY to satisfy `Ops::REQUIRED_KERNELS` - their own comment
says "Compiled, never dispatched" - and a full grep of both crates confirms
neither ever calls `Ops::decode_scores_batched` or dispatches the
`paged_decode_scores*` family at all. Their real decode-attention primitive
is `model::block::gqa_decode_step` (dispatching `attn_decode_scores`), used
by `qwen35::serve`/`qwen35moe::serve` - a structurally different, simpler
kernel family by explicit design: `qwen35::serve`'s own module doc states
`block_size == max_seq_len` so "one physical block backs one sequence's
whole KV history", chosen specifically to avoid "needing block-indirect
(scatter/gather) attention kernels" like `paged_decode_scores_wg`'s
block-table contract requires. `attn_decode_scores` has no cooperative
sibling anywhere in `docs/reference/kernels.md` (only a windowed variant,
same `@opt 2/5`, same one-thread-per-output shape) - there is nothing to
wire these two crates onto without writing a new kernel, which is Phase 5
territory (`kernels.md` §A: "does a good kernel already exist" - here the
honest answer is no), not this milestone's. Not touched; recorded here per
decision 3 rather than force-fitting a selector call that would name a
kernel family these crates do not use.

### M1.1's MoE milestone - `Op::MoeExpertLinear`, and the `kv_int8`-shaped claim that turned out true this time

`Op::MoeExpertLinear` lands scoped exactly as the recalibration table below
prescribes: the dtype/quant tier only (F32/BF16/F16 vs I8 vs Q4), mirroring
`Op::MatMul`'s `Dtype` arm, with the dense-loop-vs-compact-vs-decode-sparse
dispatch policy in `crates/model/src/moe.rs` left unmigrated - that policy is
host-synchronizing and data-dependent, and already differs by design between
models (glmdsa always compacts; qwen35moe only at `n==1`). Unlike
`Op::MatMul`, this Op has NO shape gate at any dtype: none of
`moe_linear_gated{,_i8,_q4}.wgsl` has a cooperative or register-tiled
sibling, by construction - a per-thread early `return` for a non-routed row
is only safe without a `workgroupBarrier()` in the kernel at all, so there is
no decode-vs-prefill regime to split on. F32/BF16/F16 is unconditionally
`Reference`; I8/Q4 is `PackedInt8`.

The claim this milestone's brief carried forward - "qwen35/qwen35moe's
int8-expert path never checks `caps.numeric.int8_dot`" - has the exact same
shape as the paged-attention milestone's `kv_int8` claim above, and was
investigated the same way (`kernels.md` §B: read the kernel source before
gating anything on it). This time the claim held: `moe_linear_gated_i8.wgsl`
calls `dot4I8Packed` once per weight-scale group in its inner loop, so it
genuinely is DP4A-bound, unlike `paged_decode_scores_i8_batched`'s plain
scalar bit-unpacking. `Dtype::Q4` mirrors `Dtype::I8`'s `int8_dot`
requirement exactly as `Op::MatMul`'s own Q4 arm already does, even though
`moe_linear_gated_q4.wgsl` itself unpacks nibbles with plain scalar
bit-shifts and calls `dot4I8Packed` nowhere - the SAME mismatch
`matmul_q4_dyn`/`matmul_q4_gemv` already carry against `Op::MatMul`'s Q4 arm
(this ledger's own "Q4 uses zero `dot4I8Packed`" finding above). Fixing that
mismatch is Phase 5 (M5.5) territory for both Ops alike, not re-litigated
per-Op here - mirroring `Op::MatMul`'s arm means inheriting its known
imperfection too, not quietly correcting only the new copy.

TDD: `moe_expert_linear_is_capability_only_with_no_shape_gate` was written
first and failed to compile before the variant existed (the same shape as
`Op::PagedAttention`'s own precedent - a new enum variant makes the match in
`candidates` non-exhaustive until the arm is added).
`candidates_head_is_the_default_policy` extended to cover the new Op. Full
`brain-backend-api` suite (40 tests) and `cargo clippy -p brain-backend-api
--all-targets` stay green. Landed as its own first, tight commit touching
`select.rs` (`f9a66961`), per this campaign's contention rule for that file -
a concurrent uncommitted `Op::Conv2d` change already in the working tree at
the time was set aside (`git diff` saved, file reverted to `HEAD`) before
this milestone's edit, then reapplied via a 3-way merge after the commit and
verified byte-identical to its pre-existing form. No caller in the tree
dispatches through this Op yet - migrating `crates/model/src/moe.rs` and its
per-model callers onto it is `M1.2`/later-Phase-1 territory, per the same
"seam first, migration second" split `Op::MatMul` itself already went
through.

---

## M1.1's scope, recalibrated against an exhaustive call-site map

Before touching `select.rs`, every call site making a capability/shape-gated
kernel choice for the six planned families (attention, paged attention,
softmax, conv2d, embed, MoE expert linear) was mapped exhaustively. The
finding: **not all six fit `select.rs`'s pure `candidates(op, shape, caps) ->
Vec<KernelVariant>` signature equally well**, and forcing a family that
doesn't fit produces a wrong abstraction - exactly what this campaign's own
goal ("minimize even the chance of using the wrong kernel") argues against.
Per-family verdict:

| Family | Verdict | Why |
|---|---|---|
| **Softmax** | Full fit, do first | Structurally identical to `Op::MaxAbsRow` (`WorkgroupPerOutput`/`Reference`, capability-only, no shape gate). Only two sites (`wan`, `ltxv`) duplicate the same rule; every other attention family dispatches an ungated fixed kernel - a missed win, not a bug, and the easiest, lowest-risk migration. |
| **Paged attention** | Full fit, high value | The scores half is exactly `Op::MaxAbsRow`'s shape too. Real bugs found along the way: `qwen3::serve`'s `kv_int8` branch never checks `caps.numeric.int8_dot` (unlike its own `weights_int8`/`w8_on` sibling, which does); `qwen35`/`qwen35moe` never register or reach the `workgroup_reductions`-gated cooperative scores kernel `qwen3::serve` gets, with no marker of the absence (the `Option<usize>` pattern MatMul/Conv1d use for "caller didn't register this" is missing here). |
| **Conv2d** | Partial fit | Two independent, structurally different decision trees exist (`vision::blocks::Conv` - env-var/shape/registration-driven, no `DeviceCaps` read anywhere; `vae::blocks::Builder::conv_s` - capability + shape gated, a genuine Conv1d analogue). Scope `Op::Conv2d` to the `vae::blocks` tree only; record `vision::blocks`' tree as explicitly out of scope rather than force-fitting it. |
| **MoE expert linear** | Partial fit | The dtype/quant tier (F32/BF16/F16 vs I8 vs Q4) matches `Op::MatMul`'s `Dtype` arm exactly, including the same missing-gate bug: `qwen35`/`qwen35moe`'s int8-expert path never checks `caps.numeric.int8_dot`. The dense-loop-vs-compact-vs-decode-sparse policy does NOT fit - it is host-synchronizing (a mid-layer `g.read` of routed rows) and data-dependent, not a static device-capability decision, and its policy already differs by design between models (glmdsa always compacts; qwen35moe only at `n==1`). Scope `Op::MoeExpertLinear` to the dtype/capability axis; leave the compaction policy as an explicit model-level decision, unmigrated. |
| **Attention** (dense/GQA flash) | Real debt, highest risk | The actual ladder (`flash_bidir_variant`/`flash_cross_supported`/`gqa_attn_sublayer_fwd`) is already centralized in `model::block`. The bug is the OUTER gate deciding whether to even ask the ladder: `wan::block::attn_mode`, `lfm2::Model::flash_selectable`, `sdxlunet`'s `self.coop`, and `ltxv::block::flash_self_attn`/`flash_cross_attn` each reimplement a *different* subset of the same check (`workgroup_reductions` alone; plus "ladder beat baseline"; plus "not training"; plus "head_dim <= 128") - lesson #78's exact shape ("a selection seam only reaches callers that opt in"), just for a gate instead of a kernel. Needs its own careful design pass, done last and separately, not folded into this milestone's commit. |
| **Embed** | Out of scope | Barely a selection problem - mostly dtype-only, and the one real device-capability input (`gpu.max_storage_binding_bytes()`, a byte *limit*, not a boolean/threshold) doesn't map onto `OpShape` cleanly. The one real bug found (`gpt2` dispatches a bare `EMBED` with no vocab tiling at all, unlike every other model, "safe" only because its vocab is small enough not to hit the binding-size failure `qwen3::model.rs` already documents fixing) is a one-line fix unrelated to a new `Op` variant - tracked separately, not as part of the selector-widening work. |

So M1.1 lands as: **Softmax → Paged attention → Conv2d (`vae::blocks` only) →
MoE (dtype axis only) →** the gpt2 embed-tiling fix (unrelated one-liner, but
found in the same audit) **→** the Attention outer-gate consolidation last,
as its own carefully-scoped piece of work. This is more commits than the
original plan's "one per op family" implied, because two of the six
"families" turned out to be two decisions each (fit vs no-fit).

### M1.1's Conv2d milestone - `Op::Conv2d`, scoped to `vae::blocks` only

Lands with the shape the recalibration predicted: capability + shape gated
(`RegisterTiled` requires `workgroup_reductions` via `KernelVariant::requires`,
plus BOTH `Cout >= GEMM_CONV2D_MIN_COUT` (32) and `hw >= GEMM_CONV2D_MIN_HW`
(128) - unlike `Op::Conv1d`, a 2D conv's output-position count genuinely can
be small, so there is a decode-shaped regime here to protect). Migrated
`vae::blocks`'s own `GEMM_CONV_MIN_COUT`/inline `hw >= 128` check into
`select.rs` verbatim, sweep provenance included, so the threshold lives in
the one place the decision is made. `vision::blocks::Conv`'s separate
env-var/registration-driven tree (no `DeviceCaps` read anywhere) stays
explicitly out of scope, unchanged. `brain-vae` and `brain-backend-api`
(40 tests, incl. the new gate test) green; `brain-flux2`/`brain-sdxlunet`
(downstream VAE consumers) build clean. Zero clippy warnings. Commit
`f87f85a0`.

### M1.1-moe-int8-dot-wiring - closing the last open piece of the MoE milestone, and a second recalibration on `qwen35` (dense)

The MoE milestone above scoped `Op::MoeExpertLinear` to `select.rs` only and
left `crates/qwen35moe/src/model.rs`/`crates/qwen35/src/model.rs`'s own
construction-time decision ("should this instance even build int8 weights")
unmigrated. Checked both against source rather than taking that scoping note
as license to skip verification:

**`qwen35moe` (MoE) had a real, narrower bug than the brief described.**
`Qwen35::new_impl_on`'s `q8` field (`crate::q8::Qwen35Q8`, the packed int8
bank for every routed expert's gate/up/down) was built unconditionally
whenever `i8` was requested, with no `caps.numeric.int8_dot` check and no
fp32 fallback of its own - unlike the 9 GDN/GQA mixer linears beside it,
which already self-gate via `model::ops::Weight::upload`'s own
`want.promote(&ops.caps.numeric)` call (the `weights` field's own doc already
documented this asymmetry: "F32 unless ... the device caps support the DP4A
path" for `weights`, no such clause for `q8`). This mattered for real: on a
Vulkan/wgpu GPU whose caps report no `shaderIntegerDotProduct` device feature
(`backend-vulkan`'s `int8_dot` is exactly that measured feature,
`ctx.prec.dp4a`), an unconditionally-built `q8` would still hand
`model::moe::expert_fwd_i8` a `moe_linear_gated_i8.wgsl` dispatch - the same
`dot4I8Packed`-calling, DP4A-bound kernel the already-landed
`Op::MoeExpertLinear` policy (`f9a66961`) requires `int8_dot` for. Fixed by
computing `i8_on = i8 && gpu.caps().numeric.int8_dot` once in
`new_impl_on` (mirroring `qwen3::serve::Engine::from_map_with_gpu`'s
`weights_int8`/`w8_on` pattern exactly, including the same
"print the fallback, never degrade silently" `eprintln!`) and using it for
the `q8` construction, the mixer-linear upload closure, and the `ParamStore`
role-exclusion filter that all three previously drove off the raw,
caps-blind `i8` flag.

An existing CPU-backend test (`int8_forward_completes_on_cpu_backend_with_
mixer_weights_demoted_to_fp32`) had locked in the OLD, ungated behaviour as
intentional, reasoning that `moe_linear_gated_i8.wgsl` has no
`workgroupBarrier()` and so happens to execute correctly through the CPU
JIT's software `dot4I8Packed` lowering (confirmed true - `wgsl-cpu`'s
`Dot4I8Packed` lowers to four sign-extending scalar `imul`s regardless of
`int8_dot`) even though `backend-cpu::caps` reports `int8_dot: false` for an
unrelated reason (`matmul_i8_dyn`/`matmul_i8_gemv_reg`'s multi-barrier shape,
not `dot4I8Packed`'s own correctness). That reasoning is correct about the
CPU JIT specifically but is exactly the kind of implicit, kernel-by-kernel
capability exception this campaign's Phase 1 exists to remove: `int8_dot`'s
own doc states the packed-dot kernels "execute" as a per-device fact, not a
per-kernel one, and the already-landed `Op::MoeExpertLinear` policy already
requires it unconditionally. Updated that test (now
`int8_forward_matches_fp32_exactly_on_cpu_backend_lacking_int8_dot`) and
`int8_model_excludes_quantized_names_from_the_fp32_param_store` (moved to the
default backend, since the exclusion it checks now only happens on a
capable device) to match the corrected contract, and added
`int8_moe_dispatch_is_unreachable_without_int8_dot` - the dedicated gate test
this task's brief asked for, plus a `Qwen35::moe_int8_active()` accessor so a
test can observe the gate without reaching into the private `q8` field. TDD:
both new/renamed CPU tests were confirmed RED against the pre-fix code
(reverted `i8_on` usage back to raw `i8`, re-ran) before being confirmed
GREEN after.

**`qwen35` (dense) has no MoE and no bug here at all - the brief's premise
was wrong for this crate, the same way the kv_int8/paged-attention findings
were wrong before it.** It has no `q8` field; every quantizable linear (the
12 per-layer mixer/MLP leaves) already lives on `self.weights` and goes
through the identical `Weight::upload` self-gate the qwen35moe mixer linears
use. Its own existing test suite already proves and documents this end to
end (`int8_forward_matches_fp32_almost_exactly_on_cpu_backend_full_demotion`:
"an 'int8' CPU build is actually a COMPLETE fp32 demotion", cosine >
0.999999) - a grep for the literal string `int8_dot` finding zero hits in
`qwen35/src/model.rs` reflects that the gate lives one layer down in
`Weight::upload`'s shared `promote` call, not that it is missing. Left
untouched; no changes landed in `crates/qwen35/src/model.rs` or its tests.

Full `brain-qwen35moe` suite (53 tests across `lib` + 10 integration files,
including the 7-test `model_i8_smoke` file) green on the default backend;
`cargo clippy -p brain-qwen35moe --all-targets` zero warnings. Does not touch
`crates/backend-api/src/select.rs`.

### M1.1-attn-gate - `model::block::flash_gate`, the attention outer-gate consolidation deferred to the end

The scope recalibration above named this the highest-risk piece of `M1.1` and
deferred it to its own pass. Checked against source rather than taken on
trust: the actual flash-vs-materialized ladder
(`flash_bidir_variant`/`flash_cross_supported`/`gqa_attn_sublayer_fwd`) was
confirmed already centralized in `model::block`, but the OUTER gate deciding
whether to even ASK it was reimplemented four times, each a different subset
of the same `caps.workgroup_reductions` check: `wan::block::attn_mode` (the
check alone), `lfm2::Model::flash_selectable` (plus "the ladder actually beat
the materialised baseline rung"), `sdxlunet::Rec`'s self-attention (plus "not
a training/gradient-recording pass"), and `ltxv::block::flash_self_attn`/
`flash_cross_attn` (plus `head_dim <= 128`) - lesson #78's exact shape, just
for a gate instead of a kernel: `flash_bidir_variant` itself does not read
`workgroup_reductions` (it only picks a rung by shared-memory/workgroup-size
fit), so every caller had to make that correctness check itself before
asking, and a future change to it would have had to be hunted down and
reapplied in four places.

Added `model::block::flash_gate(caps, extra) -> bool`
(`caps.workgroup_reductions && extra`) as the one shared predicate, with each
site's genuinely different extra condition kept as an explicit argument
rather than folded into a config enum (train-mode exclusion, the measured
"beats the baseline" check, the `head_dim` ceiling - forcing these into one
shared type would only move the duplication into picking which enum variant
each site needs). Migrated all four sites onto it; deleted `sdxlunet::Rec`'s
now-dead `coop` field (its only reader). `ltxv::block::flash_cross_attn` ANDs
`flash_gate` with the already-centralised, stricter `flash_cross_supported`
explicitly (shared memory + workgroup size on top of the same
`workgroup_reductions` bit), since that is a different, correct gate for the
cross family and not part of the duplication this milestone targets.

Landed as two commits (wan/lfm2/sdxlunet migrated together once their full
suites confirmed green; ltxv separately once its own, much longer, suite
confirmed green) rather than one, since the milestone's own gate held each
crate's full suite to green independently and there was no reason to block
the confirmed three on the slowest one. A new `flash_gate` unit test in
`model::block` pins the truth table at all four `(workgroup_reductions,
extra)` points. Verified: `brain-model` (153 tests, incl. the new test),
`brain-wan`, `brain-lfm2` and `brain-sdxlunet` full suites green (wan's
real-weight `dit_parity`/`gguf_import_real`/`gguf_direct_real` included); the
full `brain-ltxv` suite (41 integration test files, incl. real-weight
`dit_parity`/`av_dit_parity`/`upscale`/`vae_tiling`) green end to end; zero
clippy warnings on all five crates. `make parity`'s CPU-backend gradcheck
suite passed clean (61/61); its Vulkan-backend gradcheck suite could not be
run clean on this box - it hard-fails on `bf16_train::tests::matmul_bf16_
weight_eps_plateau` ("this harness requires a real bf16 weight" - `Weight::
upload`'s `DType::promote` gate correctly demoting BF16 to F32 on a P40,
which per this campaign's own hardware section has none), then further tests
that share the weak GPU device pool appear to hang rather than fail cleanly,
reproducing identically in two independent, isolated runs. Confirmed
unrelated to this change by dependency graph, not just by rerun: `flash_gate`
is a pure addition and every migrated call site lives in `wan`/`lfm2`/
`sdxlunet`/`ltxv`, none of which `t5`/`clip`/`sam2`/`vqgan`/`deepseekocr`/
`unet`/`restore`/`supir`/`bf16_train` (the failing set) depend on. Recorded
here rather than in `lessons.md` since the poisoning mechanism is inferred,
not yet root-caused to the level that rule expects.

Found but out of scope for this pass: `flux1`, `flux2` and `minimaxmusic3`
each duplicate the identical `caps.workgroup_reductions` check for their own
flash-attention outer gate (`flux1`/`flux2`'s `push_attention`,
`minimaxmusic3::dit::flash_attn`), not named in this milestone's four sites.
`flux1`/`flux2` reuse the SAME boolean (`self.fast`) for both this gate and
`model::block::gemm_variant`'s GEMM-tier decision, so migrating them is not a
drop-in swap the way the four named sites were - it needs the two concerns
split first. Left unmigrated; a future pass can fold them onto `flash_gate`
once that split is done.

### M1.3 - `check-kernel-selection.sh`, a workspace-wide gate, and the inventory it produced

Checked the brief's premise against source rather than following it literally:
"generalize `crates/qwen3/tests/no_kernel_names.rs`" reads as "replace it",
but that test's check 1 (banning `crate::q8::Q8`/`Lin8` INSTANCE inspection
anywhere in the crate) polices an internal-API-design invariant with no
kernel-catalogue analogue at all - a generic name-vs-catalogue gate cannot
express it and would silently drop that coverage. Left `no_kernel_names.rs`
untouched and added `scripts/gates/check-kernel-selection.sh` as a second,
complementary, workspace-wide gate (wired into `check/scripts`, which
`test/full` already depends on) rather than forcing a replacement.

**What "faster sibling" means, cross-referenced against `kernels.md` rather
than assumed**: two kernels are siblings if their names share a stem once
trailing "structural variant" words are stripped from each - the six suffix
families the milestone brief names (`_rows`/`_wg`/`_reg*`/`_tiled`/`_part`/
`_dyn`) plus two the catalogue's own naming convention already needs for the
same purpose (`_batched`, `_final`). A first, more permissive version that
stripped ANY trailing token misclassified `conv2d_dw`/`conv2d_dx` (backward
passes) as siblings of `conv2d_tiled` purely because `dw`/`dx` are also
strippable-looking tokens - caught by hand-checking the tool's own output
against the kernel sources before trusting it, the same discipline this
campaign's `kv_int8`/paged-attention and MoE recalibrations already needed.
Restricting the strip vocabulary to the eight real structural words fixed it:
19 real stem families with a genuine `@opt` spread, 23 individual "slow"
kernel names.

**What "outside a selection seam" means**: the slow kernel's identifier as
the first argument of a real dispatch call (`.step`/`.step_buf`/
`.step_sliced`/`.dispatch`), unless the call lives in `select.rs` itself or a
`KernelVariant::`/`.select(Op::`/`selector.select(`/`candidates(Op::` token
appears in the ten lines above it - the shape every real seam consumer
already has (`model::ops::Ops::bind`, `optim::Optim::coop_gradnorm`,
`qwen3::serve::Engine::rms`). A pipeline-table registration line
(`("name", kernels::NAME)`) or a `const NAME: usize = …` index declaration is
not itself a selection and is correctly never flagged - it is the identifier
comparison rule (`[A-Z][A-Z0-9_]*` only) that keeps a lowercase local
binding of the same word (`crates/model/tests/moe_compact_parity.rs`'s
`matmul: &Op` parameter, seen and fixed during the same pass) from being
mistaken for a kernel index.

**The inventory this gate produced, seeded into its own allow-list, 44
rows over 7 kernel names and 20 files** (every row carries its own reason in
the script; not reproduced verbatim here): `matmul`/`matmul_dw`/`matmul_dx`/
`rmsnorm`/`layernorm` bare-dispatched in roughly a dozen model/training
crates (`deepseek2`, `qwen35`, `qwen35moe`, `toyseq2seq`, `toypid`, `toymoe`,
`toyautoencoder`, `kronos`, `fincast`, `mimi`, `chronos2`, `qwen3omnimoe`,
`qwen3tts`, `gpt2`) that have never been migrated onto the `MatMul`/`RmsNorm`/
`LayerNorm` `Op`s that already exist - filed as Phase 1 M1.4 / Phase 5
backlog, not fixed by this gate. `matmul_dw`/`matmul_dx` specifically have NO
`select::Op` at all yet (a gap Phase 5's own family table, M22, does not
itemise) - filed the same way, flagged here so it is not lost. `qwen3::
model.rs`'s own `lora_fwd`/`proj_bwd` still bare-dispatch `MATMUL` too - B7's
migration scoped only `forward_steps`/`decode_steps`/`run_batched_steps`/
`head_steps`, not LoRA or backward, so this is pre-existing, not a new
regression. `decode_softmax` in `glmdsa`/`gpt2`'s own incremental-decode path
has no cooperative sibling wired through `select::Op` either - the same
paged-attention triad this ledger's M0.2 baseline already flagged as the
campaign's top target, now confirmed present in two more model crates.
`minimaxmusic3::discriminator`'s `conv2d` dispatch is outside `Op::Conv2d`'s
deliberately narrow scope (`vae::blocks::Builder::conv_s` only), the same
category as `vision::blocks::Conv`'s already-documented exemption.
`crates/model/tests/tensor_parallel.rs`'s raw `matmul`/`matmul_dw`/
`matmul_dx` steps are a dp/shard-parity test harness by design, never through
`model::ops::Ops`.

Every OTHER kernel this gate's stem analysis found a faster sibling for
(`clip_coef`, `conv2d_gd`, `conv_act`, `conv_bias`, `gn_stats`,
`layernorm_dx`, `ln_stats`, `matmul_gemv`, `matmul_rows`, `paged_decode_apply`,
`paged_decode_scores`, `paged_decode_scores_batched`, `prelu_bwd`,
`flash_attn_bidir`) had ZERO unallowed dispatch sites - already fully behind
an existing seam (`optim::Optim::coop_gradnorm`, `Op::MatMul`, `Op::
PagedAttention`) or never bare-dispatched at all.

Mutation-verify: removed the `matmul`/`crates/gpt2/src/model.rs` allow-list
row, confirmed the gate turned RED listing exactly `gpt2/src/model.rs`'s 6
bare `MATMUL` dispatches with `matmul_reg`/`matmul_reg2`/`matmul_reg3` named
as the faster siblings, then restored the row and confirmed GREEN again. The
gate also fails on a STALE row (one that no longer matches any real
violation), verified the same way, so the allow-list can only ever track
reality rather than merely grow. `bash scripts/gates/check-kernel-selection.sh`
green; `check-scripts.sh` (syntax/orphan/absolute-path),
`check-no-doc-citations.sh` and `check-doc-links.sh` green for the new file.
`check-env-docs.sh`/`check-no-perf-numbers.sh`/`check-arch-names.sh` were
already red on this tree before this change, from unrelated pre-existing
findings (none in any file this milestone touched) - left as-is, not this
milestone's scope. No Rust source changed, so no crate's test suite,
`clippy`, `parity` or `gradcheck` is affected.

### M1.4 - Closed the `step_buf` blind spot; M1.3's inventory has no drop-in row to add

Checked the brief's second half against source before building it, per the
AGENTS.md rule this campaign keeps re-invoking: "add an upgrade row for every
drop-in-qualifying pair M1.3's gate inventory found" reads as "there are rows
to add", but M1.3's own inventory is seven kernel names
(`matmul`/`matmul_dw`/`matmul_dx`/`rmsnorm`/`layernorm`/`decode_softmax`/
`conv2d`), and every one of them fails at least one of `gpu_core::upgrade`'s
own four bars once read against its actual WGSL:

- **`rmsnorm` -> `rmsnorm_rows`**: different `Params` struct (`d_model,
  seq_len` vs `d, rows, eps`) and `rmsnorm_rows.wgsl`'s own header states the
  agreement is `max_abs 3.3e-6` because "the reduction order differs" - fails
  bar 1 (contract) and bar 2 (bit-identical) on its own words.
- **`layernorm` -> `layernorm_rows`**: same `Params` this time, but
  `layernorm_rows.wgsl`'s header documents a DIFFERENT algorithm (the shifted
  one-pass form, forced by the CPU JIT's one-barrier limit) against
  `layernorm.wgsl`'s textbook two-pass - "agreement... is checked in
  `bench_layernorm`", i.e. tolerance, not bit-identity. Fails bar 2.
- **`decode_softmax` -> `decode_softmax_batched`**: different `Params`
  (`n_heads, t, cap` vs `batch, n_heads, cap`) and an extra `seq_lens` binding
  - a batched rewrite, not a same-contract thread-count change. Fails bar 1.
- **`matmul`/`matmul_dw`/`matmul_dx` -> their `_reg`/`_reg2`/`_reg3` siblings**:
  contract and accumulation order both hold (`select.rs:323`'s own comment:
  "the `matmul_reg*` family accumulates strictly in increasing `k`", confirmed
  by reading `matmul_reg.wgsl`'s chunk loop - `gk = c*BK+kk` visits `0..K-1` in
  order, same as `matmul.wgsl`'s serial loop, so no reassociation). What fails
  is bar 3: `gpt2::model::linear_kernel`'s own measured threshold (`m < 8` before
  BLK 128x128 tile wins) and `dx_kernel`'s (`m < 128 || k < 128`) are exactly
  the regimes `gpt2::model::decode_at`'s bare `MATMUL` dispatch runs at
  (`m = 1`, single-token decode) - the naive kernel is not a defect there, it
  is the FASTER choice at that shape, so a blanket redirect would regress every
  decode step. The kernel this seam could add would have to win at `m = 1`
  too, and by construction the 128x128-tile kernel cannot.
- **`conv2d` (`minimaxmusic3::discriminator`) -> the register-tiled sibling**:
  same shape-dependence, confirmed by `select.rs`'s own `Op::Conv2d` test
  (`narrow_cout`/`narrow_hw` -> `Reference`, `wide` -> `RegisterTiled`) -
  already a policy `select::candidates` owns, not a constant-win drop-in.

The common shape is not a coincidence: every "fast" sibling in this inventory
only wins in a shape regime, which is precisely what `select::Op` (not
`gpu_core::upgrade`) exists to arbitrate, and duplicating that shape policy as
a second copy inside `upgrade::UPGRADES` would be exactly the "one
implementation" rule violation this campaign already corrects elsewhere. The
seven names stay migration targets for an explicit `Op::MatMul`/`Op::LayerNorm`
/`Op::RmsNorm` call-site migration (already the pattern `model::ops::Ops`
provides) or a Phase 5 kernel rewrite - not this seam. Recorded here as a
correction to M1.3's own filing, not a failure of this milestone: the premise
was checked, found not to hold, and the real defect (the `step_buf` blind spot
the milestone's first half named) was fixed instead.

**The real fix**: `gpu_core::upgrade`'s shape-specialised rows (the
`matmul_gemv`/`matmul_i8_gemv` `MREG` ladders) could not resolve a bucket from
`Gpu::step_buf`, because `apply` had no `params` to read - the uniform lives in
a caller-owned buffer the seam cannot see into. Added `Gpu::step_buf_shaped`
(both the native and wasm facades) alongside the unchanged `step_buf`: a caller
that already holds the values it wrote into its own uniform buffer hands them
back as a `shape: &[u32]` probe, reaching `upgrade::apply`'s existing
`Some(params)` path exactly as `step`/`step_sliced` already do. `step_buf`
itself is untouched - same signature, same `None` probe, same fallback to the
registered kernel.

TDD: `crates/gpu-core/tests/gemv_reg_upgrade_step_buf.rs` went RED first
(`step_buf_shaped` did not exist) then GREEN, on real Tesla P40 hardware via
`Gpu::kernel_times()` (the same per-pipeline device-timing table
`BRAIN_PROFILE` prints) as the oracle for which PHYSICAL kernel actually ran:
`step_buf` alone stays on `matmul_gemv` at every `m`; `step_buf_shaped` reaches
the exact same `MREG` bucket `step` would pick (`m=1`->`MREG=1`, `m=3`->
`MREG=4`, `m=32`->`MREG=32`); the two dispatch paths agree bit-for-bit at
`m in {1,5,17,32}`; `BRAIN_NO_KERNEL_UPGRADE=1` pins `step_buf_shaped` back onto
`matmul_gemv` the same way it already does for `step`. Full `brain-gpu-core`
suite (24 test binaries, 69 lib unit tests including `upgrade::tests`) green on
real hardware; `cargo clippy --all-targets` clean for `brain-gpu-core` and
`brain-backend-api`. No `select.rs` change, so nothing else in this campaign's
most-contended file was touched.

### M2.1 - `paged_flash_decode`, a corrected gate, and a measured regression worth recording

Wrote `crates/kernels/wgsl/paged_flash_decode.wgsl`: one workgroup per
`(sequence, head)` walks that sequence's block table in `BC = 8`-key tiles,
each tile's `head_dim` dot product split across `LANES = 8` threads (on the
SAME flat top `paged_decode_scores_wg`'s own sweep records for "8 and 4"),
running online max/sum and folding the softmax-weighted V straight into a
per-lane accumulator - no `scores`/`probs` buffer at all, unlike the
`paged_decode_scores{,_wg}` -> `decode_softmax_batched` ->
`paged_decode_apply_batched` triad it sits beside. Five top-level
`workgroupBarrier()`s (one query stage + four per key tile) exceed the CPU
JIT's one-barrier-per-body limit, so it is `@cpu no`, a GPU-only sibling; the
three-stage path stays registered as the CPU/reference implementation behind
`Op::PagedAttention`, untouched - wiring the fused kernel in through that
selector is M2.4's job, not this one's, and `select.rs` was not touched this
milestone.

**The plan's "bit-comparable" gate does not hold - checked against source
before writing the test, per this campaign's own rule, not taken on trust.**
`rmsnorm_rows`'s own precedent (this file's `block.rs`: "64 partial sums fold
in a different order, agreeing to ~3e-6") and `flash_attn_causal_gqa`'s own
gate against its materialized reference (`1e-3` absolute error, not
`assert_eq`) both establish that a reassociated online-softmax reduction is
never bit-exact against a two-pass exact-max reference: the triad computes
one exact max over the whole row before a single un-rescaled exp/sum pass,
this kernel rescales its running sum once per tile. The test
(`crates/model/src/paged.rs::flash_tests::paged_flash_decode_matches_batched_
triad`) asserts a `1e-3` maxabs bound instead, matching that precedent.
Measured maxabs on this box: `2.3841858e-7`, identical on `BRAIN_DEVICE=gpu`
(wgpu) and `BRAIN_DEVICE=vulkan` - both this kernel's GPU backends, `@cpu no`
deliberately excluding the CPU JIT. `gradcheck` unaffected (forward-only
path, no new `Op` variant).

**Two defects a first draft shipped with, caught before landing.** (1) The
first `Params` carried both `n_kv_heads` and a separate `kv_stride` field,
values that are always mutually derivable (`kv_stride == n_kv_heads *
head_dim` at every real call site checked) - the kernel body never actually
read `kv_stride`, a dead uniform. Dropped it; the pool's row stride is
computed from `n_kv_heads * head_dim` exactly as `flash_attn_causal_gqa.wgsl`
already does, per the milestone's own instruction to copy that kernel's
`Params` shape. (2) The first tile size (`BC = 16`, `LANES = 4`) sized shared
memory at ~16.9 KiB - OVER WebGPU's guaranteed 16 KiB
`maxComputeWorkgroupStorageSize` floor that `flash_attn_causal_gqa` sits
exactly AT. Corrected to `BC = 8`/`LANES = 8` (~8.8 KiB): same measured
coalescing optimum, half the tile footprint, comfortably portable.

**Measured delta against the M0.2 baseline shape - a regression, published
honestly rather than assumed away.** `crates/qwen3` was mid-edit by a
concurrent session (a selector-migration refactor leaving `qwen3::serve`
uncompilable for the duration of this milestone), so `qwen_bench
flash-decode` - added to `crates/qwen3/src/bin/qwen_bench.rs` as this
milestone's reproducer, `qwen_bench flash-decode [seq] [reps]` - could not be
run through the qwen3 binary itself; the identical dispatch code was run
through a throwaway `brain-gpu-core` integration test instead (deleted after
use, no dependency on the blocked crate) to avoid blocking this milestone on
someone else's unrelated WIP. At Qwen3-0.6B's real decode-head shape
(`n_heads=16, n_kv_heads=8, head_dim=128`) and M0.2's own `seq_len=cap=512`
steady-state regime: at `batch=1` the triad (`paged_decode_scores_wg` +
`decode_softmax_batched` + `paged_decode_apply_batched`) took 0.41 ms against
the fused kernel's 0.83 ms; sweeping `batch` (the concurrent-decode-batch
regime continuous batching actually runs at) to 8/32/128 converges to the
fused kernel taking consistently ~1.8-1.9x the triad's time at every size
(0.55x/0.55x/0.53x throughput ratio), the triad reaching up to 115% of the
measured DRAM roof (cache-resident at this size) against the fused kernel's
61%. Root cause, not just the number: this design dispatches only
`batch * n_heads` workgroups (e.g. 2048 at `batch=128`), each serialising
`ntiles = cap / BC = 64` barrier-synced tile iterations one after another;
the triad's scores kernel instead dispatches one independent workgroup per
*score* (`batch * n_heads * cap / 16` of them - over a million at
`batch=128`), so it hides the SAME global-memory latency behind far more
parallelism than a single-kernel, tiled-online-softmax design can generate at
this shape. Eliminating the `scores`/`probs` buffer traffic (this design's
whole rationale) does not pay for the parallelism given up to get it, on
this hardware, at this shape - this is exactly the kind of finding decision 4
names as a legitimate, published, non-blocking outcome of measuring before
trusting the audit's architectural inference: the kernel is a correct,
GPU-only sibling as specified, but M2.4 (wiring it behind the selector) must
not treat this as a drop-in win without re-measuring against a design that
raises this kernel's own occupancy (e.g. a split-key-then-combine two-pass
shape) - flagged here so that work is not repeated blind.

**Commit**: one (the kernel + catalogue regen + the correctness test);
`qwen_bench flash-decode` lands in the same commit since it is this
milestone's own published reproducer, not a separate change.

### M2.2 - `paged_flash_decode` int8-KV twin and bf16-storage tier

Two siblings of M2.1's fused kernel, gated on correctness only (the
milestone's own gate is "cosine/rel_l2 vs the fp32 fused kernel", not a perf
target) - **occupancy was not re-measured at either shape**, so M2.1's own
caution ("this kernel's tiling strategy loses to the triad's parallelism at
this hardware/shape") is inherited unchanged by both new siblings, not
independently confirmed or refuted; M2.4 must weigh all three the same way.

`paged_flash_decode_i8.wgsl`: a genuinely new physical kernel, not a
`dtype_variant` of the fp32 one - `pool_k`/`pool_v` become 4-int8-per-`u32`
packed pools plus per-`(token, kv-head)` `scales_k`/`scales_v`, dequantized
once while staging a tile into shared memory (everything downstream is the
unmodified fp32 body). Same scale/round-clamp scheme
`paged_decode_scores_i8_batched`/`paged_decode_apply_i8_batched` already use.
8 storage buffers - exactly the WebGPU guaranteed floor. This is the worst
case to fuse against: the int8 path has no `_wg` cooperative sibling at all
today, so the replaced reference is three dispatches, each re-reading or
re-writing the `[batch, n_heads, cap]` scores/probs slab.

`paged_flash_decode`'s bf16 tier needed no new kernel source at all: `pool_k`/
`pool_v` already index with the bare identifier `kernels::template::
dtype_variant` requires, so two CHAINED `dtype_variant` calls (`pool_k` first,
then `pool_v` over that call's own output) produce the templated source - the
same mechanism `paged_decode_scores_batched#pool_k=bf16`/
`paged_decode_apply_batched#pool_v=bf16` already use for the split pair,
applied twice here because this kernel reads both pools in one dispatch.
`@dtype` updated to `f32|bf16` with a `@tpl` block documenting the chain.

**Gate, both variants**: `crates/model/src/paged.rs::flash_tests` compares
each variant against the plain fp32 FUSED kernel (not the three-stage triad),
at `rel_l2 < 0.01` / `cosine > 0.99` - the same bound `qwen3::serve`'s own
`int8_kv_scale_and_bytes_match_a_host_oracle` gates `kv_int8`'s serving
tolerance at. Measured on this box (wgpu, Tesla P40, GQA shape with a
scrambled shared block-table pool, `seq_lens` straddling the kernel's `BC=8`
tile size): int8 `rel_l2 = 0.0036`, `cosine = 0.999993`; bf16
`rel_l2 = 0.0022`, `cosine = 0.999998` - both comfortably inside the gate,
and bf16's smaller error than int8's is exactly the expected ordering given
bf16 keeps 7 explicit mantissa bits against int8's 7-bit symmetric range.

**Verified via a throwaway harness, documented rather than hidden.**
`crates/qwen3/src/serve.rs` was mid-edit by a concurrent session for this
milestone's entire duration (real compile errors - `CachedSelector` vs
`Arc<dyn KernelSelector>`, a `tuned_i8` field removed from `Engine` mid-edit -
not just slow contention), which blocks `brain-model`'s own test target
(`brain-gradcheck`, a hard dev-dependency, itself hard-depends on
`brain-qwen3` with no feature gate to route around). The measurements above
came from running the identical dispatch code through a throwaway
`brain-gpu-core` integration test (no dependency on the blocked crate,
deleted after use) - the exact same workaround M2.1's own entry already
recorded for the same reason. `crates/model/src/paged.rs`'s real tests were
written, reviewed against the actual kernel/template source, and left in
place as the durable gate; they were not run through `cargo test -p
brain-model` itself before this entry was written, because that build was not
possible during this milestone's window. Whoever next builds `brain-model`
clean should treat a red result here as a real regression report, not
assume it away.

**Commit**: two (`paged_flash_decode_i8` + its test; then the bf16 `@tpl`
header change + its test), per the milestone's own "one per variant" split.

### M2.3 - `paged_flash_prefill`, `flash_attn_causal_gqa`'s tiling ported onto the paged pool

`crates/kernels/wgsl/paged_flash_prefill.wgsl`: one workgroup owns `BR = 64`
causal query rows of ONE sequence's prefill chunk, per (head, query-tile) -
`flash_attn_causal_gqa.wgsl`'s own BR-tiled, lane-split-head_dim tiling (the
register-spill fix that kernel's header already derives in full - `q`/`o` in
`array<f32,32>` per lane, not `array<f32,128>` per thread), ported from a
dense `[B*T,...]` K/V buffer onto `paged_flash_decode.wgsl`'s block-table
addressing. **Not a repeat of decode's own tiling strategy** (one query per
workgroup, `LANES=8`) that M2.1/M2.2's own entries left un-re-measured for
occupancy - this is a different, already-registered `@opt 4` shape, so that
caution does not transfer here. No `scores`/`probs` buffer at all: the
three-stage triad it replaces is dispatched by `qwen3::serve::
run_batched_steps` once per prefill CHUNK with `bsz` = chunk length (checked
against that function's source, not assumed), so its scratch is exactly the
`[nh,N,N]` shape `Engine::from_map_with_gpu`'s own scratch-sizing comment
names once `cap` grows to cover a whole chunk.

Same tape as `paged_flash_decode`/the triad (`Params`, `q`/`pool_k`/
`pool_v`/`block_tables`/`seq_lens`/`ctx`): `qwen3::serve::prefill` already
builds `seqlens[i] = start+i+1` and duplicates one block table across every
row of a chunk (source-checked: one prefill dispatch is always one
sequence), so M2.4's wiring needs no host-side buffer changes. Two contracts
the kernel states explicitly, both following from that same fact: every row
in a workgroup's tile shares one physical block table (read through the
tile's first row), and `seq_lens` is non-decreasing across a tile (so the
workgroup's largest live-key count, and how many K tiles it visits at all,
is its last row's own value) - the same assumption `flash_attn_causal_gqa`
already makes implicitly (row i's boundary IS i there; here it is data, so
the kernel says so).

**The plan's "token-for-token" gate does not literally hold, for the same
reason M2.1's own entry already recorded and this milestone inherits
unchanged (checked against source before writing the test, not taken on
trust): a reassociated online-softmax reduction, rescaling once per
`BC=8`-key tile, is never bit-exact against the triad's exact-max-then-
single-pass reference.** Gated at the same `1e-3` absolute-error bound
`paged_flash_decode_matches_batched_triad` uses
(`crates/model/src/paged.rs::flash_tests::
paged_flash_prefill_matches_batched_triad`), comparing the attention CONTEXT
the triad already produces during prefill (not a full forward's logits -
everything downstream of attention is unchanged here, so a matching `ctx` is
the direct, sufficient proof). Scenario: one sequence, `start=17`
already-cached tokens, a `cc=130`-row chunk spanning three `BR=64` query
tiles (exercises the tile-boundary and causal-early-exit logic, not just one
full tile), GQA (`n_kv_heads=2` of `n_heads=4`), a scrambled (reversed)
block-table permutation. Measured maxabs on this box: `4.172325e-7`,
identical on `BRAIN_DEVICE=gpu` (wgpu) and `BRAIN_DEVICE=vulkan` - same order
of magnitude as M2.1's own `2.3841858e-7`.

GPU-only by construction (`@cpu no`, 3 top-level `workgroupBarrier()`s per
key tile, over the CPU JIT's one-barrier limit) - the three-stage path stays
registered as the CPU/reference implementation; this is an additional GPU
sibling. `select.rs` was not touched this milestone; wiring behind
`Op::PagedAttention` and shrinking `Scratch::{scores,probs}` is M2.4's job.
No perf/occupancy measurement was taken (the milestone's own gate is
correctness only, matching M2.1/M2.2's precedent of leaving the wired-in
measurement to M2.4); `gradcheck` is unaffected (forward-only path, no new
`Op` variant).

**Verified via a throwaway harness, documented rather than hidden - the
identical situation M2.1's and M2.2's own entries already recorded, now a
third time with the SAME root cause.** `crates/qwen3/src/serve.rs` was
mid-edit by a concurrent session for this milestone's entire duration (the
same `CachedSelector` vs `Arc<dyn KernelSelector>` / `tuned_i8`-field compile
errors M2.2's entry already quotes, unchanged), which blocks `brain-model`'s
own test target the same way (`brain-gradcheck` hard-depends on
`brain-qwen3`, no feature gate to route around). The measurement above came
from running the identical dispatch code through a throwaway `brain-gpu-core`
integration test (no dependency on the blocked crate, deleted after use).
`crates/model/src/paged.rs::flash_tests::
paged_flash_prefill_matches_batched_triad` was written, reviewed against the
actual kernel source, and left in place as the durable gate; it was not run
through `cargo test -p brain-model` itself before this entry was written,
because that build was not possible during this milestone's window either.
Whoever next builds `brain-model` clean should treat a red result here as a
real regression report, not assume it away.

**Commit**: one (the kernel + catalogue regen + the correctness test).

---

### M2.4 - Wire in `Op::PagedAttentionFused`, shrink the scratch, and re-measure both regimes before trusting either

**The plan's own premise ("wire the fused kernels through `Op::PagedAttention`'s
policy") does not literally hold, checked against source before touching
it.** `Op::PagedAttention` is deliberately scoped to the SCORES half only
(its own doc comment says so), and `model::block::paged_scores_variant` - the
one existing caller - matches that Op's selector result against
`WorkgroupPerOutput` specifically; splicing a third candidate into that same
list would have silently changed what a SCORES-only caller sees. Added a
SEPARATE, new Op instead - `Op::PagedAttentionFused` plus
`KernelVariant::FusedFlash` - scoped to the whole-triad-vs-single-fused-dispatch
decision, keyed on `(k, dtype)` where `k` names the regime (`0` = decode,
independent sequences, no shared block table; `1` = causal-chunk prefill, one
sequence's block table shared by every row) since the two are different
physical kernels answering the same shape signature in different call-site
semantics, not points on one shape gradient. `candidates()` unit-tested
directly (`paged_attention_fused_only_offers_the_fused_kernel_at_causal_
chunk_f32`).

**Re-measured before wiring anything in, per the "not an unconditional win"
warning M2.1-M2.3's own entries already left standing.** Added `qwen_bench
flash-prefill` (mirrors `flash-decode`'s own harness: the exact triad
`qwen3::serve::prefill` dispatches per chunk vs `paged_flash_prefill`, at
Qwen3-0.6B's real head shape) and swept `start` (already-cached prefix) /
`cc` (chunk length) on this box, both wgpu and vulkan:

| start | cc | triad | `paged_flash_prefill` | speedup |
|---|---|---|---|---|
| 0 | 64 | 0.8846 ms | 0.5471 ms | 1.62x |
| 0 | 512 | 9.4144 ms | 1.5742 ms | 5.98x |
| 512 | 512 | 26.1953 ms (wgpu) / 23.6814 ms (vulkan) | 2.9814 ms (wgpu) / 2.6178 ms (vulkan) | 8.79x / 9.05x |
| 1536 | 512 | 67.9631 ms | 5.6198 ms | 12.09x |

A real, growing win as `start` grows - exactly the shape expected from the
root-cause difference the M2.1 finding already named for the sibling kernel
(the triad's SCORES/APPLY kernels walk every `cap` slot per row regardless of
live length; the fused kernel walks only `start+cc`), except here the SAME
mechanism helps instead of hurting because prefill's `BR=64`-tiled,
lane-split-head_dim shape (M2.3's own, ported from the already-registered
`flash_attn_causal_gqa`) generates enough parallelism per workgroup that
eliminating the `scores`/`probs` traffic is pure upside. **Decode's fused
kernels (M2.1/M2.2) were NOT re-measured and were NOT wired in** - their own
entries' measured regression is a kernel-shape fact (worse parallelism,
independent of `cap`), not a shape-crossover this milestone's own new data
could plausibly overturn, so `Op::PagedAttentionFused`'s `k = 0` arm stays
`Reference`-only at every dtype. This is the "killed, not forced" outcome
Phase 5's own rubric names as a legitimate result, applied one phase early.

**Wired into `qwen3::serve::run_batched_steps`** via a new `causal_chunk:
bool`, threaded through `run_batched`/`run_batched_submit`/
`run_batched_greedy`/`steps_for_profile`. `Engine::prefill` and
`Engine::score_positions` pass `true` (both checked against source: one
sequence, `seqlens[i] = start+i+1`, one block table duplicated across every
row of the chunk - exactly `paged_flash_prefill`'s own stated contract).
Every decode call site passes `false`. `Engine::spec_decode`'s verify-forward
structurally qualifies too (same one-sequence-causal-chunk shape) but is
deliberately left on the triad this milestone - noted inline as a follow-on,
not re-litigated here.

**A real correctness bug caught before it shipped, not after.** The first
version of the `Scratch::{scores,probs}` shrink (see below) gated on
`kv_int8` alone - "fp32 KV always gets the fused prefill path now." It does
not: `FusedFlash` also requires `caps.workgroup_reductions`, true on every
GPU backend measured above but false on the CPU JIT, which is `qwen-
serving-perf-gate.sh`'s own default backend. On that device the dispatch
correctly falls back to the triad (the selector's own capability gate), but
the shrunk scratch would have stayed sized for the fused kernel's zero need -
an out-of-bounds device write on any causal chunk longer than `max_batch`.
Fixed by deriving the shrink decision through the IDENTICAL
`Op::PagedAttentionFused` selector call the dispatch site makes
(`paged_attn_scratch_bytes` takes `fused_prefill_available: bool`, computed
once at `Engine::from_map_with_gpu` via `DefaultSelector.select(Op::
PagedAttentionFused, ..., &caps)`), so the two can never drift apart again.
Recorded as a general rule in `.agents/rules/kernels.md` (F.7b), not just
fixed locally.

`Scratch::{scores,probs}` - `b*nh*cap`, this campaign's own audit finding as
"the single largest serving scratch buffer" - shrinks to decode's own worst
case (`max_batch*n_heads*cap`) whenever the fused path is reachable, dropping
the `max_prefill^2*n_heads` `[nh,N,N]` term the old, unconditional `max(...)`
formula always paid. Pinned by `paged_attn_scratch_shrinks_once_the_fused_
prefill_path_replaces_the_triad` at a representative shape (`max_batch=128,
max_prefill=512, n_heads=16, cap=2048`): 128 MB -> 32 MB, exactly 4x. An
int8-KV engine, or any engine on a device without `caps.workgroup_
reductions`, gets no reduction - `paged_attn_scratch_shrinks_only_when_
fused_prefill_is_actually_reachable` pins that directly against a hand-built
CPU-shaped `DeviceCaps`, no CPU backend needed.

**Also fixed, as a genuine prerequisite, not scope creep: the concurrent
`CachedSelector<DefaultSelector>` -> `Arc<dyn KernelSelector>` migration that
had left `crates/qwen3/src/serve.rs` uncompilable for M2.1/M2.2/M2.3's ENTIRE
duration (each of those three entries records hitting the identical compile
errors and working around them with a throwaway harness).** The migration's
own doc comments already fully specified the target state; only three call
sites had not been updated to match (a deleted `tuned_i8` field two call
sites still referenced, and the `selector` field's construction still using
the pre-migration type). Restored `tuned_i8` as the plain `HashMap` field
`Self::mm8` already expected, and wrapped the `DefaultSelector` construction
in `Arc` - completing exactly what was already designed, not redesigning
anything. `brain-qwen3` and `brain-model` build and test clean as a result,
closing the "confirm `cargo test -p brain-model` is green" item the "Not yet
done" section below has carried since M2.1.

**Verified.** Full `brain-qwen3` suite: 104 passed, 1 ignored (`#[ignore]`d
throughput benchmark) - no failures across several full runs; the
pre-existing NVIDIA-driver teardown SIGSEGV-on-exit flake `backend-vulkan.md`
already documents was seen once, always AFTER every test's own `ok` line,
not caused by this change. `make gradcheck`: 21 suites, 0 failed (forward-only
change, no new backward-differentiable `Op` variant, but run anyway since the
milestone touches a live kernel dispatch). `make check/scripts`'
`check-kernel-selection.sh` and `check-no-perf-numbers.sh` both clean against
every file this milestone touched (the doc-comment numbers in the table
above live in this ledger, not in source - `check-no-perf-numbers.sh` only
scans `docs/**/*.md` and source narration, not `.agents/`).

**A second existing gate caught a real mistake before it shipped, the same
"checked, not assumed" pattern as the scratch-sizing fix above.**
`qwen3/tests/no_kernel_names.rs`'s own `migrated_forward_paths_never_hand_
pick_a_gemm_kernel` bans a literal reference to the selector's return enum
anywhere inside `run_batched_steps`'s own source text - not scoped to GEMM
names specifically, a blunter rule than its own doc implies. The first
wiring inlined the `Op::PagedAttentionFused` selector call directly in
`run_batched_steps`, tripping it. Fixed by factoring the call into
`model::block::paged_attention_fused`, mirroring `paged_scores_variant`'s
own already-established shape exactly (that function lives outside
`run_batched_steps` for the identical reason) - `run_batched_steps` now only
calls it by name. Re-verified: `no_kernel_names.rs` (3 passed),
`check-kernel-selection.sh` (unaffected - `paged_flash_prefill` has no stem
sibling in the catalogue either way), full `brain-qwen3` `--lib` and
`--tests` (104 + 19 binaries, 0 failed) after the move.

**`make parity`/`make test`, run last, at the same time this ledger entry
was being written on a box already running several other sessions' own
builds against the SAME checkout.** `scripts/gates/parity-gate.sh`'s CPU-
backend gradcheck stage - the identical suite `make gradcheck` above already
covers - passed clean a second time. Its Vulkan-backend stage hit `backend-
vulkan.md`'s own documented pre-existing NVIDIA-driver hang (distinct from
that file's SIGSEGV-on-exit entry, the sibling failure mode the same section
already names: "one run hung instead, needing SIGKILL"): the `unet`/`vqgan`
gradient tests - unrelated diffusion models, no qwen3/attention code in
their path - ran 35+ minutes burning a full CPU core with BOTH GPUs at 0%
utilisation the whole time (`nvidia-smi`, sampled repeatedly), the exact
signature that file's own entry describes. Killed by hand (`SIGKILL`); the
script's own `run()` wrapper correctly recorded that one stage FAIL and
continued to the next. `cargo build --release` (workspace) passed clean.
`make test` and the remaining `parity-gate.sh` stages (model FD suites,
qwen-serve CPU-backend, TTS codec) were still compiling when this entry was
written - confirmed genuinely progressing, not stalled (dozens of live
`rustc` children with real, growing CPU time; the workspace's full
release+LTO test-binary count climbing steadily), just slow: this box was
running several concurrent sessions' own `cargo` invocations against this
SAME checkout for the whole of M2.4's window, and cargo's own build-directory
lock serialises overlapping work across ALL of them, not just within one
session. Left running in the background; whoever next has a quiet box should
let them finish and treat a real failure there (not a repeat of the killed
Vulkan hang) as a genuine regression report.

**Commits**: seven - `backend-api: add Op::PagedAttentionFused` (`select.rs`
alone, per this campaign's own file-contention rule), `model: cover
KernelVariant::FusedFlash in Ops::matmul's dispatch-count match` (the
resulting exhaustive-match fix), `backend-api: drop bare perf numbers from
Op::PagedAttentionFused's doc comment` (a `check-no-perf-numbers.sh`
follow-up), `qwen3: wire Op::PagedAttentionFused into serve, shrink
Scratch::{scores,probs}` (the milestone's own change, including the
prerequisite compile fix), `docs: record the selector/scratch-sizing rule
M2.4 caught` (F.7b), `model, qwen3: move the M2.4 fused-attention selector
call out of run_batched_steps` (the `no_kernel_names.rs` fix above).

### M3.2 - Device admission head, and `PagedDecoder::admit_greedy`/`admit_topk`

`qwen3::serve::Engine` kept a SECOND, host-only copy of the LM head
(`head: Vec<f32>`) purely to run `model::hostmath::matvec_par` against it at
admission - the one-time first-token pick after `prefill` - even though the
same head weight was already uploaded to the device (`lin_weights`) and
already dispatched every decode step via `head_steps`/`submit_greedy_head`/
`submit_topk_head`. Deleted the field and `Engine::logits`'s host matvec;
`logits` now writes its one hidden row into `sc.xn_final` and reuses
`head_steps` for the device matmul instead. Admission itself no longer calls
`logits` at all: added `Engine::admit_greedy` (writes the row, reuses
`submit_greedy_head`, reads back one index) and `Engine::admit_topk` (reuses
`topk_from_hidden`, reads back at most `TOPK_CAPACITY` candidates) so
admission never ships a `[vocab]` block to the host either for the pick or
for the values.

`PagedDecoder::logits` stays in the trait (`qwen3::eval`, `generate_greedy`
and `spec_decode` still want a raw per-row logits vector), but the scheduler
no longer calls it for admission - it calls two new trait methods,
`admit_greedy`/`admit_topk`, that DEFAULT to a plain host argmax / host sort
over `logits`'s vector (byte-for-byte the code this replaced, just moved out
of `Scheduler::step_inner` and into the trait), so `qwen35`/`qwen35moe` -
which still have no device head at all (M3.4) - keep their exact prior
behaviour with no changes to either crate. `qwen3::serve::Engine` overrides
both to the device paths above; this is what actually closes the milestone's
"sorts a whole `[vocab]` vector on the host" finding at `model/src/serve.rs`
for the one decoder that has a device top-k to move it to.

Two pre-existing unit tests (`device_head_argmax_matches_the_host_head`,
`split_argmax_matches_the_host_head_at_large_vocab`) called the per-row
`logits` in a loop immediately before `greedy_from_hidden` read the SAME
batch back out of `sc.xn_final` - safe under the old host-only `logits`, but
`logits` now writes row 0 of that same scratch buffer, so the batched read
had to move before the per-row loop; reordered rather than left racing.
Added `admission_head_matches_a_true_host_matvec_within_tolerance`: unlike
the two tests above (which compare two device computations sharing the same
`logits_dev` values and so can assert exact equality), this one compares
`admit_greedy`/`admit_topk` against an INDEPENDENT host matvec built straight
from the checkpoint map in the test, never through `Engine::logits` - a
tiled-GEMM-vs-scalar-dot-product reduction-order difference is real here, so
the assertion is a per-index value tolerance, not index equality, per this
campaign's own gate wording. Re-verified: full `brain-qwen3` `--lib`
(105 passed, 1 ignored) and `brain-model`/`brain-qwen35`/`brain-qwen35moe`
`--lib`, zero clippy warnings across all four crates.

`crates/qwen3/src/serve.rs` carried unrelated uncommitted M3.1 work (the
`prefill` chunk-readback consolidation and its test) in the working tree
throughout this milestone - a different concurrent session's in-progress
change to the same file, not this milestone's own. Split by hunk
(`git apply --cached` against a hand-trimmed patch) so this milestone's two
commits touch only what M3.2 actually changed, leaving M3.1's hunks
untouched and unstaged for that session to commit itself; re-verified both
in combination (build/clippy/`--lib`) and with M3.1's hunks stashed out
(build/clippy/`serve::` tests) to confirm neither depends on the other.

**Commits**: two - `model: move admission's greedy/top-k pick onto
PagedDecoder` (trait + scheduler, a pure refactor via default trait methods -
no behaviour change for any decoder), `qwen3: delete the host admission
head, reuse the device head for it too` (Engine + tests).

### M3.4 - `qwen35::serve` gets a device head, and `prefill`'s per-token read is gone

`qwen35::serve::Engine` is architecturally NOT `qwen3::serve::Engine` - it is
the deliberately single-sequence, per-token-dispatch, correctness-first
engine its own module doc names (`Qwen35::run_decode_step` has no batch
dimension at all: GDN's recurrent state and the flat per-block GQA cache are
both `n = 1` shaped), so "batched prefill" and "batched greedy" in the
audit's literal sense - one dispatch across many prompt tokens, or across
many sequences - are NOT buildable without rearchitecting the model's own
decode primitive, which is out of this milestone's scope and is exactly the
"Deliberately deferred" list the module doc already carries (chunked prefill,
multi-sequence GPU batching). What the audit's finding actually named as
concrete defects, and what this milestone fixed instead, checked against
source per this campaign's own rule:

- **No device head at all** (confirmed: `Engine::head: Vec<f32>` +
  `hostmath::matvec_par`, used for BOTH admission and every decode step) -
  even though the SAME head weight was already resident on the device via
  the model's own `ParamStore` (`run_forward`'s training-path head epilogue
  already dispatches `MATMUL` against `self.w(cfg.head_weight())` at full
  model scale), so the host copy was a pure duplicate, exactly `qwen3`'s
  M3.2 shape. Added `Qwen35::head_logits_dev`/`head_argmax_dev`/
  `head_topk_dev` (device `MATMUL` + the shared `argmax_part`/`argmax_final`
  split-reduction + `topk_extract_step` - all three already-cataloged
  kernels, newly REGISTERED in `qwen35::model::pipelines()`, never
  hand-written); `Engine::forward_batched_greedy`/`forward_batched_topk` now
  chain `decode_one`'s returned `DeviceBuffer` straight into them without an
  intermediate host readback, and the `PagedDecoder::admit_greedy`/
  `admit_topk` overrides (added per M3.2's own trait seam) upload the
  admission hidden row and reuse the same two methods, so admission never
  ships a `[vocab]` block to the host either. `Op::ArgMaxRow`'s
  `SplitReduction` kernels are capability-free (no `caps` gate in that arm -
  `backend_api::select`'s own doc), so dispatching them unconditionally
  (this crate carries no `KernelSelector` of its own) is correct at every
  vocab size, including the 29-token tiny test config (`argmax_part.wgsl`'s
  own `end = min(start + chunk, n)` bounds a chunk index past `n` to an
  empty range, so the excess chunks contribute `-inf` and never win).
- **A per-token `read` in `prefill`** (confirmed: `gpu.read(&h, d)` on every
  loop iteration, discarding every result but the last). Fixed by chaining
  `run_decode_step`'s device buffer across the loop and reading back exactly
  once, after it ends - `qwen3::serve::Engine::prefill`'s own M3.1 shape
  ("submit every step, read back once"), ported at token granularity instead
  of chunk granularity since this engine has no multi-token batched dispatch
  to chunk over.
- **A sequential host loop for `forward_batched_greedy`**: still a host loop
  (multi-sequence GPU batching is the out-of-scope item above), but each
  iteration's OWN head projection + sampling pick is no longer a host round
  trip - see the device-head bullet.

TDD: `prefill_reads_back_exactly_once_regardless_of_prompt_length` (new,
mirrors `qwen3`'s `prefill_submits_scale_with_chunks_not_with_token_count`)
confirmed RED against the pre-fix code (`got 3` readbacks for a 3-token
prompt) before the fix and GREEN after, on the default backend.
`forward_batched_topk_matches_an_independent_host_matvec_within_tolerance`
(new) replays the same steps through a SEPARATE `Qwen35::step`-driven
instance and an independent host `matvec_par` + sort, never reusing any
device kernel this milestone added, and matched both value (1e-3 tolerance)
and id at every one of `k=5` candidates. The two pre-existing
`scheduler_decode_matches_step_{cpu,default_backend}` tests (bit-exact
greedy decode vs `qwen35::sample::generate_kv`, which computes logits via
the SAME independent host `matvec_par`) stayed green through the whole
change on both the CPU JIT and the default (wgpu) backend, which is the
strongest existing evidence the new device head's reduction order agrees
with the host reference. Full `brain-qwen35` `--lib` (49 passed, 1 ignored)
and `--test serve` (4 passed) green; `cargo clippy -p brain-qwen35
--all-targets` zero warnings; `scripts/gates/check-kernel-selection.sh`
exits clean (the new `argmax_part`/`argmax_final`/`topk_extract_step`
dispatches are the CATALOGUE's fast siblings, not the slow `argmax_row` the
gate polices, so no allow-list row was needed; `matmul` in
`crates/qwen35/src/model.rs` was already an allow-listed M1.4/Phase-5
backlog row before this milestone and covers the new head dispatch too).

`scripts/gates/qwen35-perf-baselines/qwen35-resident-int8-cpu48-gpu2.json`
exists locally on this box, but it is NOT a `qwen35::serve::Engine` baseline
- its own `notes` field says "qwen35 int8 GGUF two-card resident", i.e. it
measures `crates/qwen35/src/int8_gguf_resident.rs` (a completely different,
disk-streamed, dual-GPU, int8-quantized code path this milestone never
touches), not the single-GPU fp32 `Engine` this milestone changed. No
benchmark binary in the tree drives `qwen35::serve::Engine`'s decode tok/s at
all (`qwen35_bench`/`qwen35_decode_profile` both drive `int8_gguf_resident`/
`stream` instead), so there is no baseline this milestone's change could be
measured against, and the numeric comparison this milestone's gate asked for
is skipped rather than fabricated against a mismatched artifact - a future
`Engine`-specific decode-throughput benchmark is the real prerequisite.

**Commits**: two - `qwen35: add Qwen35::head_{logits,argmax,topk}_dev, the
device-head machinery` (registers the three already-cataloged kernels in
`pipelines()` and adds the methods, self-contained and unused by anything
yet), `qwen35: port qwen3's device head onto serve::Engine, fix prefill's
per-token read` (Engine + tests).

### M4.1 - Fused QKV and gate/up projections in `qwen3::serve`

Concatenated `attn.{wq,wk,wv}` (`[hq+2*hkv, d]`) and `mlp.{gate,up}`
(`[2*ff, d]`) at engine WEIGHT-LOAD time (`Engine::from_map_with_gpu`), not
at the on-disk checkpoint importer: `W:[out,in]` is row-major, so
concatenating along `out` is exactly concatenating the flat row-major
arrays end to end, read straight from the same host weight map the split
leaves were already read from. `import.rs`/`gguf_import.rs`/
`decoder_param_list` are untouched - a checkpoint on disk still has five
split tensors, and the fused `attn.wqkv.weight`/`mlp.gateup.weight` names
exist only in `Engine::lin_weights`, at runtime. One GEMM now replaces
three (Q/K/V), one replaces two (gate/up); `run_batched_steps` narrows the
wide fused output back into the compact `q_pre`/`k_pre`/`v`/`gate_pre`/`up`
buffers QK-norm/RoPE/KV-append/`swiglu_fwd` already require via
`concat_split.wgsl` - the existing kernel, per `qwen35moe::model`'s own
kernel-reuse note that `region_copy` cannot do this job (it requires
src/dst to share one `row_stride`/`off`; `concat_split` gathers a wide
strided row into a fresh compact buffer). No new kernel, as the milestone
required.

**Gate**: `fused_qkv_and_gateup_are_bit_identical_to_split` (new) proves
this exactly, no tolerance - not against a host reference (a tiled device
GEMM and a scalar host loop genuinely reduce in a different order, per
`admission_head_matches_a_true_host_matvec_within_tolerance`'s own doc,
so that comparison would prove nothing about bit-identity here), but
against the split path run through the SAME device kernel (`Engine::mm`)
this engine dispatched before this milestone, over the three/two original
unconcatenated weight matrices and the fused dispatch's own real prefill
activations. Passed on first write against the implementation (built
alongside the test, not strictly red-then-green, given how much of the
milestone was investigating which existing kernel could do the narrowing
step at all - see below). `cargo test -p brain-qwen3 --lib`: 106 passed on
the GPU/default backend (includes every pre-existing forward/decode parity
test: `batched_serving_matches_reference`, `chunked_prefill_matches_whole`,
`decode_window_path_matches_the_single_step_reference`,
`warm_prefill_is_identical_to_cold`, `spec_decode_matches_greedy`, both
int8-weight and int8-KV variants). `cargo clippy -p brain-qwen3
--all-targets`: zero warnings.

CPU backend (`BRAIN_DEVICE=cpu`) surfaces two failures in `serve::tests`
that do NOT belong to this milestone: checked by isolating this change's
own diff from unrelated uncommitted work sitting in the same working tree
(another in-flight milestone's decode/prefill submit-batching WIP) and
building/testing the isolated result standalone, both
`causal_chunk_fp32_kv_dispatches_the_fused_kernel_not_the_triad` and
`decode_step_submits_are_not_one_per_metadata_write` reproduce identically
against a clean, unmodified `git HEAD` checkout with none of this
milestone's changes present. The first is paged-attention-fused kernel
selection (this milestone never touches `Op::PagedAttentionFused` or its
selector); the second is `run_batched_submit`/`submit_greedy_head` issuing
two separate `gpu.submit()` calls per decode step (this milestone adds
steps to the vector each already submits, never an extra `submit()` call
of its own) - pre-existing, unrelated, left alone.

**Measured per-kernel table delta** (`qwen_bench serve`, Qwen3-0.6B shape,
2x Tesla P40, against the M0.2 baseline): decode (`serve 1 20 512`) went
590 -> 646 dispatches, 18.14 -> 17.77 ms (-2.0%, 55 -> 56 rows/s); prefill
(`serve 128 20 512`) went 786 -> 758 dispatches, 132.18 -> 125.73 ms
(-4.9%, 968 -> 1018 rows/s). Per this campaign's own §E requirement, the
mechanism is NOT reduced memory traffic - `concat_split` reads and writes
the full fused-output width (at this shape, `2*b*(hq+2*hkv)` extra words
at decode/QKV, more than the `2*b*d` words the fused GEMM saves by
reading its input activation once instead of three times), so it shows up
as a genuinely new, non-trivial line (decode: 140 calls, 1.03 ms, 5.6% of
the pass, only 0.8% of its own memory roof; prefill: 140 calls, 2.03 ms,
1.6% of the pass, 50.6% of roof). The actual win is dispatch count and
per-call roofline efficiency on the DOMINANT, weight-bandwidth-bound
GEMM/GEMV itself: at decode `matmul_gemv` drops 196 -> 112 calls
(7.72 -> 7.09 ms) at 80.1% -> 87.1% of roof; at prefill
`matmul_reg3_splitk` drops 196 -> 112 calls (34.42 -> 29.53 ms) at
31.8% -> 37.1% of roof and its `dw_splitk_reduce` fold drops
196 -> 112 calls (13.73 -> 9.78 ms) in lockstep. A wider `N` per dispatch
streams the identical total weight bytes more efficiently and the engine
pays for `concat_split` out of dispatches saved elsewhere, not out of a
smaller memory footprint - both regimes net a real, if modest (prefill)
to small (decode), whole-pass improvement, so this is a kept win, not a
killed hypothesis.

**Commits**: two - `qwen3: fuse Q/K/V and gate/up projections into two
GEMMs (M4.1)` (`Engine::from_map_with_gpu` fused-weight construction,
`concat_split_step`/`WQKV`/`WGATEUP`, the two dispatch-site rewrites, and
the bit-identity test), this ledger entry.

### M4.2 - Fused QK-norm + RoPE + KV-append in `qwen3::serve`

Checked against source first, per this campaign's own rule: the plan's literal
"KV_APPEND x2" reads as both K's and V's append dispatch, but V never goes
through RMSNorm or RoPE - only K does. So the real fusable region is the FIVE
dispatches that actually share the same per-head row end to end (`rms(q)`,
`rms(k)`, `ROPE_PAGED(q)`, `ROPE_PAGED(k)`, and K's own `KV_APPEND_B`), not six;
V's append is a separate row and stays a separate dispatch. Two new kernels
(`qknorm_rope_fused.wgsl`, `qknorm_rope_append_fused.wgsl`) collapse that into
ONE dispatch for Q (norm+RoPE) and ONE for K (norm+RoPE+fp32 paged append):
one workgroup per `(batch, head)` row, the SAME single-`workgroupBarrier()`
reduction shape as `rmsnorm_rows` - RoPE's `(m, m+half)` pair is drawn from the
SAME row RMSNorm just normalized, so after the one reduction barrier every
thread re-reads its own pair from global memory (not a `var<function>` array
sized off a runtime `head_dim` - the exact anti-pattern named in
`docs/performance/overview.md`), applies norm-scale and rotation together, and
- for K - writes the rotated value to BOTH `sc.k` (still needed by
`Engine::calibrate_kv` and test fixtures) and its paged-pool slot in the same
store. Gated on the queried `caps.workgroup_reductions` exactly like
`Engine::rms`'s own cooperative arm - a device without it (`backend-cpu`'s own
doc: "the split-at-barrier JIT mis-executes the workgroup-cooperative
reduction kernels") keeps the original unfused `rms`/`ROPE_PAGED`/`KV_APPEND_B`
sequence, so nothing regresses there. The int8-KV branch fuses only norm+RoPE
for K (2 dispatches to 1): its own append (`APPEND_I8_CLIPPED`) does a
whole-row absmax reduction into a packed `u32` pool, a different shape that is
NOT folded in this milestone.

**Gate**: `qk_norm_rope_fused_is_bit_identical_to_the_unfused_pair{,_kv_int8}`
(new) prove exact bit-identity against the unfused `rms` -> `ROPE_PAGED` (->
`KV_APPEND_B` for K, fp32-KV variant only) pair, run against the SAME
`q_pre`/`k_pre` inputs `prefill`'s last layer actually fused - not a
tolerance check, since normalizing then rotating the same values in the same
order is not a reassociation. The fp32-KV test also reads back the paged pool
at each row's real `(block, offset)` slot and checks it against the same
reference, so the new append address arithmetic is checked, not just the
norm+RoPE math `sc.k` alone would cover. `cargo test -p brain-qwen3 --lib`:
109 passed (108 before this milestone's two new tests + 1 ignored,
unchanged), on the GPU/default backend, including every pre-existing
forward/decode/calibration parity test. `cargo clippy -p brain-qwen3
--all-targets`: zero warnings.

CPU backend (`BRAIN_DEVICE=cpu`), isolated from unrelated concurrent WIP
sitting in the same working tree the same way M4.1's own ledger entry
describes: 44 passed, 1 pre-existing failure
(`causal_chunk_fp32_kv_dispatches_the_fused_kernel_not_the_triad`, confirmed
identical against a clean HEAD checkout with none of this milestone's changes
present - `Op::PagedAttentionFused` kernel selection, untouched by this
milestone) - `decode_step_submits_are_not_one_per_metadata_write`, the SECOND
pre-existing failure M4.1 recorded, does not even exist on this isolated
checkout, confirming it belongs entirely to that unrelated concurrent WIP, not
to M4.1 or M4.2.

**Measured per-kernel table delta** (`qwen_bench serve`, Qwen3-0.6B shape,
2x Tesla P40, isolated build - this milestone's own diff on top of the M4.1
commit, with no unrelated concurrent WIP mixed in - mean of 5 runs each):
decode (`serve 1 20 512`) went 646 -> 562 dispatches (-13.0%), 17.39 -> 17.28 ms
mean (-0.6%, ~58 rows/s either way - flat, inside this box's own ~8% run-to-run
noise band); prefill (`serve 128 20 512`) went 758 -> 674 dispatches (-11.1%),
126.32 -> 125.25 ms mean (-0.8%, 1013 -> 1022 rows/s). Both regimes drop
EXACTLY 84 dispatches (28 layers x 3 collapsed dispatches per layer,
independent of row count - the mechanism is dispatch-count, not per-row work,
so it shows up identically in both regimes). Per this campaign's own §E
requirement: the fused region's OWN device time drops hard (prefill:
`rmsnorm_rows` + `rope_paged` + K's share of `paged_kv_append_batched` was
~1.72 ms pre-fusion; `qknorm_rope_fused` + `qknorm_rope_append_fused` +
the QK-norm-free `rmsnorm_rows` remainder is ~1.33 ms post-fusion, a ~23%
cut in the region's own device time from no longer writing the normalized
value out and reading it back for RoPE, then writing THAT out and reading it
back again for the append) - but that region is only ~1.4% of the whole
124-127 ms prefill pass, so the whole-pass number moves by a similar small
amount, not by the region's own percentage. This is the same shape M4.1's own
entry found: a real, measured, non-fabricated win in dispatch count and
per-kernel device time, translating to a modest (prefill) to flat (decode,
inside noise) whole-pass effect because QK-norm/RoPE/KV-append were never the
dominant cost here - `matmul_gemv`/`matmul_reg3_splitk` and the paged-attention
kernels are. Kept, not killed: the dispatch-count and per-kernel-time deltas
are real and reproducible: not fabricated, but also not overstated as a
whole-pass win larger than what was actually measured.

**Commits**: one - `qwen3: fuse QK-norm + RoPE + KV-append in qwen3::serve
(M4.2)` (the two new kernels, `Engine::qk_norm_rope`/`qk_norm_rope_append`,
the `run_batched_steps` call-site rewrite, and the two bit-identity tests),
this ledger entry.

### M4.3 - Fuse RMSNorm with activation quantization in `qwen3::serve`

Checked against source first, per this campaign's own rule: the plan's
"three reads, four on the int8 path" is a description of the WHOLE per-layer
shape (`quant_once` fires four times per layer - `xn1`, `ctx`, `xn2`, `h` -
each a `max_abs_row` -> `quant_pack` pair), not a claim that every one of
those four is preceded by an `rms` write. Only two are: `ln1` -> `xn1` and
`ln2` -> `xn2`; `ctx` (attention output) and `h` (SwiGLU output) are
quantized activations that were never RMSNorm's output, so they are out of
this milestone's own title and untouched. For the two that ARE, `Engine::
linear`'s `Weight::I8` arm never reads the `x` parameter it is handed (it
reads only the pre-quantized `i8_scratch`) and `w8_on` is a single
engine-wide tier (`Engine::from_map_with_gpu`), so whenever `self.i8_scratch`
is `Some` the fp32 value `rms` wrote to `xn1`/`xn2` had NO reader at all -
`max_abs_row` then `quant_pack` re-read it twice more purely to throw it
away. Confirmed by grep across `serve.rs`: `xn1`/`xn2` have exactly three
uses each (the `rms` write, `quant_once`'s read, and `Self::linear`'s call,
whose `I8` arm ignores the buffer it's handed) and the one test that DOES
read `xn1`/`xn2`'s real fp32 content (`fused_qkv_and_gateup_are_bit_identical_
to_split`) builds an all-fp32 engine, never exercising this tier.

New kernel `rmsnorm_quant_fused.wgsl` (one workgroup per row, 3 barriers,
`@cpu no` like `softmax_rows.wgsl`'s own multi-barrier cooperative shape)
folds `rmsnorm_rows` + `max_abs_row` + `quant_pack` into ONE dispatch that
never writes the fp32 intermediate at all: stage 1 is `rmsnorm_rows`'s own
sum-of-squares reduction verbatim; stage 2 recomputes `v = x[c]*inv*w[c]`
(the exact expression `rmsnorm_rows` would have written) to fold a row-wide
abs-max into `sx[row]`, never touching a `d`-wide buffer; stage 3 recomputes
`v` once more to pack `xq[row, :]`, `quant_pack`'s own arithmetic. No
`var<function>` array sized off the runtime `d` (the anti-pattern `qknorm_
rope_fused.wgsl` already names) - recomputing `v` from `x`/`inv`/`w` a second
and third time trades cheap, cache-warm ALU for never allocating a
runtime-sized register array and never touching a `d`-wide buffer more than
`rmsnorm_rows` itself already does. `Engine::rms_quant` dispatches it when
`self.i8_scratch.is_some() && self.caps.workgroup_reductions`, else falls
back to the unfused `Self::rms` + `Self::quant_once` pair unchanged (an
all-fp32 engine, or a device without cooperative reductions, where `xn1`/
`xn2`'s fp32 value IS still the real result).

**Gate**: `rms_quant_fused_is_bit_identical_to_the_unfused_triad` (new, RED
before the kernel/dispatch existed, GREEN after) proves exact bit-identity -
not a tolerance check, since `v` is recomputed with the identical expression
and operand order every time, which IEEE754 guarantees reproduces the same
bits. Dispatches `Engine::rms_quant` directly on synthetic non-degenerate
input rather than reading `i8_scratch` back after a full `prefill`, because
`I8Scratch::sx` is ONE buffer SHARED across every K-width a layer quantizes
(`xn1`'s `d`, `ctx`'s `hq`, `xn2`'s `d` again, `h`'s `ff`) - a real forward
overwrites it several times per layer, so its state after `prefill` reflects
whichever call happened LAST in program order (`h`'s `ff`-width quant), not
`xn2`'s; an earlier draft of this test read `sx` back after `prefill` and
failed for exactly that reason - a test bug, not an implementation bug,
caught by the mismatch being two orders of magnitude off from anything the
kernel could plausibly produce. `cargo test -p brain-qwen3 --lib`: 111
passed (110 before this milestone + 1 new), GPU/default backend, including
every pre-existing forward/decode/int8 parity test
(`int8_weights_track_fp32`, `int8_kv_close_to_fp32`, both `qk_norm_rope_
fused_is_bit_identical_to_the_unfused_pair{,_kv_int8}`, `fused_qkv_and_
gateup_are_bit_identical_to_split`). `cargo clippy -p brain-qwen3
--all-targets`: zero warnings. `make kernels-table/check`: green (439
kernels, the new one's `@cpu`/`@gpu`/`@opt`/`@quant` fields cross-checked
against its own barrier count and shared-memory use).

CPU backend (`BRAIN_DEVICE=cpu`): 108 passed, the SAME two pre-existing
failures M4.1's and M4.2's own ledger entries already recorded and traced to
unrelated concurrent WIP (`causal_chunk_fp32_kv_dispatches_the_fused_
kernel_not_the_triad`, `decode_step_submits_are_not_one_per_metadata_write`)
- this milestone's own `rms_quant` gate never fires on this backend at all
(`workgroup_reductions` is false there), so it cannot be their cause.

**Measured per-kernel table delta** (`qwen_bench serve ... i8w`, Qwen3-0.6B
shape, 2x Tesla P40, isolated build - `git stash` held this milestone's own
diff aside, baseline measured, popped, rebuilt, re-measured, so the only
delta between the two runs is this milestone's own commit): dispatch count
drops from 786 to 674 (-14.2%) at BOTH regimes identically (28 layers x
4 collapsed dispatches per layer: `rmsnorm_rows`+`max_abs_row`+`quant_pack`
x2 occurrences -> `rmsnorm_quant_fused` x2, independent of row count - same
"mechanism is dispatch-count, not per-row work" shape M4.2 already measured).
Decode (`serve 1 20 512 i8w`) went 13.41 -> 13.31 ms (-0.7%, 75 rows/s either
way - inside this box's own run-to-run noise band at this precision), total
device-busy time 15.3 -> 14.5 ms (-5.2%). Prefill (`serve 128 20 512 i8w`)
went 119.46 -> 118.21 ms (-1.0%, 1071 -> 1083 rows/s). Per this campaign's
own §E requirement: the fused region's OWN device time is the real
mechanism, not the whole-pass number - at prefill, `rmsnorm_rows` + `max_
abs_row` + `quant_pack`'s combined share of the ln1/ln2 occurrences was
~1.98 ms before fusion; `rmsnorm_quant_fused` alone is 0.9 ms after, a ~54%
cut in the fused region's own device time from never writing the fp32
intermediate and only ever having ONE dispatch's worth of launch/uniform/
bind-group overhead instead of three. That region is only ~1.6% of the whole
118-119 ms prefill pass (attention - `paged_decode_apply_batched` +
`paged_decode_scores_wg` - and `matmul_i8_dyn` are ~89% of it), so the
whole-pass number moves by a correspondingly small amount - the same shape
M4.1's and M4.2's own entries already found and the same honest framing:
kept, not killed, a real and reproducible dispatch-count and per-kernel-time
win that this milestone does not overstate as more than what was measured.

**Commits**: one - `qwen3: fuse RMSNorm with int8 activation quantization in
qwen3::serve (M4.3)` (the new kernel, `Engine::rms_quant`, the two
`run_batched_steps` call-site rewrites, and the bit-identity test), this
ledger entry.

### M5.6 - MLA/DSA/GDN family: one real defect fixed (`topk_mask`), ten kernels checked against real config dims and correctly rated

The table's "5 + 6 @opt-2" count for this family resolves to exactly 11
kernels in `docs/reference/kernels.md`: `mla_scores`, `mla_index_scores`,
`mla_bwd_dk_pass`, `mla_bwd_dk_rope`, `topk_mask` (glmdsa's GLM-5.2
MLA/DSA indexer) and `gdn_decay_mask_bwd`, `gdn_decay_scale_bwd_last`,
`gdn_state_decay_bwd_dscale`, `gdn_ut_bwd_dattn0`, `gdn_ut_bwd_dtmat`,
`gdn_ut_step` (`model::gdn`'s Gated-DeltaNet backward, used by
qwen35/qwen35moe training only). Per this campaign's own discipline
("a finding is a hypothesis until checked against source"), each kernel's
actual Params-bounded reduction axis was checked against the real shapes
this repo ships (`GlmConfig::glm5_2()`: `n_heads=64`, `qk_nope_head_dim=192`,
`qk_rope_head_dim=64`, `index_n_heads=32`, `index_head_dim=128`,
`block_size=4096`; `Qwen35Config::qwen38_27b()`:
`linear_num_value_heads=48`, `linear_key_head_dim=linear_value_head_dim=128`;
`model::gdn::gdn_chunk_size` caps the chunk length at 64 for any `T`)
rather than assumed from the `@opt 2` label alone.

**`topk_mask`: a real defect, fixed.** Its dispatch gave one THREAD the
entire causal row (`b,s`): an outer serial loop over every key `t` in
`0..T`, each iteration paying its own `O(s)` causal-rank count - genuinely
`O(T^2)` on a single thread, with the row's worst case (`s=T-1`) alone
setting the whole dispatch's wall time while every other invocation sat
idle. Every `t` in a row is independent of every other `t`, so that outer
loop was serialising work that was already embarrassingly parallel.
Rewired to one thread per `(b,s,t)` cell (dispatch `bsz*T*T`, the same
`(b,h,i,j)`-style decomposition `mla_scores.wgsl` already uses) - bit-
identical output, verified against an independently-written host oracle
and the existing `indexer.rs` suite (all-pass-equals-dense, sparse-
restricts-attention, distillation, training) staying green unchanged on
both backends. Commit `8edd5ca9`.

**The other ten: checked, not force-fixed, per §F.4/F.6's discipline that a
correctly-rated kernel is a legitimate finding too.**

- `gdn_decay_mask_bwd`, `gdn_ut_step`, `gdn_ut_bwd_dattn0`,
  `gdn_ut_bwd_dtmat` all loop over `c_len` (or `i<=c_len-1`), capped at 64
  at every real config this repo ships, WHILE already dispatching
  `bhc*c_len` (or `bhc*i`) independent threads - tens of thousands of
  threads at real GDN scale, each doing a serial reduction of at most 64
  steps. Their own kernel-header doc already argues this is the correct
  tier for that shape; re-checking the real numbers confirms it: going
  cooperative here would add a `workgroupBarrier()` to shrink an
  already-tiny 64-step serial tail while the independent-thread count
  (already in the tens of thousands) is nowhere near the bottleneck. No fix
  applied.
- `gdn_decay_scale_bwd_last` has the same small `c_len<=64` reduction but a
  genuinely SMALL thread count (`threads=bh`, 48 at real scale) - real
  under-parallelisation, but the total work is `bh*c_len <= 48*64 = 3072`
  multiply-adds, trivial regardless of how it is scheduled; a dispatch this
  small is dominated by fixed per-dispatch overhead, not by how its handful
  of FLOPs are spread across threads. No fix applied.
- `gdn_state_decay_bwd_dscale` is the one GDN kernel whose own header
  comment ("`dk`/`dv` are tens to low hundreds ... matching every other GDN
  reduction's tier") does not hold at real scale: its loop is over
  `dk*dv = 128*128 = 16384` at `qwen38_27b()`'s real dims, while its thread
  count is only `bh = 48`. This is a genuine remaining defect of the same
  shape M5.1's norm-backward cooperative rewrites target (few, large,
  independent reductions) - identified but NOT fixed in this pass; filed as
  a follow-up rather than force-fit in the time this milestone had, per
  this campaign's "record it, do not force it" rule.
- `mla_scores`/`mla_index_scores` loop over `nope+rope` (256) /
  `index_head_dim` (128) respectively - small, bounded reductions - while
  already dispatching `bsz*H*T*T` (`mla_scores`) or `bsz*T*T`
  (`mla_index_scores`) independent threads, tens of millions at
  `block_size=4096`. Correctly rated; going cooperative would multiply an
  already-enormous independent-output count by a workgroup for no latency
  win, the same reasoning that rules out `mla_bwd_dk_pass` below.
- `mla_bwd_dk_pass`/`mla_bwd_dk_rope` loop over `T-j` (up to 4096) and
  `H*(T-j)` (up to 262144) respectively - genuinely large, AND the number of
  independent outputs is already enormous (`bsz*H*T*nope` /
  `bsz*T*rope`, tens of millions). A "cooperative one-workgroup-per-output"
  rewrite (the pattern that fixes a FEW large reductions) would multiply an
  already-saturating output count by a workgroup and make dispatch overhead
  worse, not better - this is the same shape that rules out `mla_scores`
  above, confirmed by the arithmetic, not assumed. The real fix is
  algorithmic: `d_k_pass[j,dn] = sum_i>=j d_scores[i,j]*q_pass[i,dn]` is a
  masked GEMM in disguise (`bmm`/`bmm_acc` already exist and `model::gdn`
  already uses them for an analogous batched contraction), and MLA has no
  flash-style backward the way GQA does (M5.2's family). Wiring MLA's
  backward onto a tiled GEMM or a flash-style algorithm is bigger than this
  pass's kernel-tiling scope - filed as a follow-up, not attempted here.

**Gate**: TDD (RED confirmed against two distinct plausible mistakes in the
`topk_mask` rewrite - a missing per-thread stride, then an off-by-one on
the causal boundary - before GREEN), `cargo clippy -p brain-glmdsa
--all-targets` zero warnings, `docs/reference/kernels.md` needed no
regeneration (the shipped kernel's `@what`/`@how`/`@opt` are unchanged from
before - only its thread-to-output mapping changed). **Commits**: one
(`8edd5ca9`, `topk_mask` only - the `gdn_state_decay_bwd_dscale` and MLA
backward-GEMM follow-ups above are unbuilt, recorded for a future pass to
pick up rather than left undocumented).

---

## Not yet done

Phase 0 is closed. Phase 1 is in progress per the recalibrated scope above.
**Phase 2 (M2.1-M2.4) is closed.** Decode's fused kernels (M2.1/M2.2,
`paged_flash_decode{,_i8}` + the bf16 tier) are correct, GPU-only siblings
registered in the kernel catalogue but deliberately NOT live in
`qwen3::serve` - measured (M2.1) and re-confirmed by the same reasoning
(M2.4) to regress against the triad on this hardware at every batch size /
dtype, so `Op::PagedAttentionFused` never offers them; a future design that
wins occupancy (M2.1's own "split-key-then-combine" suggestion) would need
its own fresh measurement, not a resurrection of these two. Causal-chunk
prefill's fused kernel (M2.3, `paged_flash_prefill`) IS live: wired behind
`Op::PagedAttentionFused` in `qwen3::serve::run_batched_steps` (M2.4),
measured a real, growing speedup as cached-prefix length grows (M2.4's own
table), and `Scratch::{scores,probs}` - the campaign's own audit-named
largest serving scratch buffer - shrinks 4x at a representative shape
whenever it is live. `brain-qwen3`/`brain-model` build and test clean (M2.4
also closed the concurrent-migration compile break that had blocked
M2.1/M2.2/M2.3's own `cargo test -p brain-model` runs for their entire
duration). **Phase 4 (M4.1-M4.3) is closed** - fused QKV/gate-up, fused
QK-norm+RoPE+KV-append, and fused RMSNorm+int8 activation quant, each a
kept (not killed) real dispatch-count and per-kernel device-time reduction
with a correspondingly modest whole-pass effect at this hardware/shape,
per §E. Phases 3, 5-8 remain, as structured in the plan. Track sub-milestone
status against the approved plan; update this section as each phase closes,
recording the measurement that proved it - a number nothing checks is a
number that silently goes stale (`AGENTS.md`'s own rule, restated here
because a multi-phase campaign is exactly where it erodes).
