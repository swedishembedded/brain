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

---

## Not yet done

Phase 0 is closed. Phase 1 is in progress per the recalibrated scope above.
Phases 2-8 remain, as structured in the plan - Phase 2 (fused paged
attention) is next by measured priority per M0.2's real numbers, not merely
the audit's architectural inference. Track sub-milestone status against the
approved plan; update this section as each phase closes, recording the
measurement that proved it - a number nothing checks is a number that
silently goes stale (`AGENTS.md`'s own rule, restated here because a
multi-phase campaign is exactly where it erodes).
