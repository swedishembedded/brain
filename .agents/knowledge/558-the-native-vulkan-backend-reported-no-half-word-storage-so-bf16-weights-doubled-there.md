<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 558. The native Vulkan backend reported no half-word storage, so bf16 weights doubled there

**Lesson:** a capability flag decides where a weight lands (`Ops::lands` -> `DType::promote`), so a backend that
under-reports one silently widens every weight of that tier: twice the bytes, and not the tier a model or a test asked
for. The storage-tier `#w=bf16`/`#w=f16` kernel variants decode packed words with core WGSL integer arithmetic, so they
run wherever WGSL runs; `backend-vulkan` never set the BF16 tier and left F16 `Absent` without `shaderFloat16`, and
`ArchDesc::numeric_view` reported `f16_storage` only for a tier exactly at `Storage`, so a device with real f16 hardware
(`Native`) widened f16 weights too. The view is now `holds()` (a native tier holds its bytes; fast arithmetic still
needs a measurement) and the Vulkan backend reports both storage tiers, as `backend-wgpu` does.

## Evidence (2026-11-22)

- GH200 through Vulkan: `head_ce` (3 tests: "the head must stay at the tier under test", F32 for BF16) and
  `qops_tier_coverage` ("bf16 must not be promoted away", then "f16 must not be promoted away") failed; the real
  Inkling 2 checkpoint, loaded through `upload_preserved` (which keeps the stored tier), ran its bf16 products through
  the same backend and matched the f32 reference, so the kernels themselves were fine.
- After: the whole Inkling 2 campaign list passes on Vulkan (`--test-threads=1`, see #300 for the concurrent
  shared-device race), `bf16_roundtrip` and `qops_frozen_lora_grad` included.
- Pinned by `backend-api` `arch::tests::a_tier_that_executes_natively_also_holds_its_bytes` and
  `tests/arch_view_agrees.rs` `vulkan_holds_half_words_unlike_the_old_formula`.
