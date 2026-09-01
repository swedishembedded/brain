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

---

## Not yet done

Phases 0 (remaining: M0.1 profiler fix, M0.2 baselines, M0.3 harness contract,
M0.4 debt sweep) through 8, as structured above. Track sub-milestone status
against the approved plan; update this section as each phase closes, recording
the measurement that proved it - a number nothing checks is a number that
silently goes stale (`AGENTS.md`'s own rule, restated here because a
multi-phase campaign is exactly where it erodes).
