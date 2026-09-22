<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 35. The "2x resident overhead" was real, but it was wgpu's, not the hardware's

§14 and `crates/qwen3/src/q8.rs` both stated the fact as a property of
non-ReBAR Pascal: "each storage buffer carries ~2x resident overhead." That
was measured correctly but attributed too broadly - it is a property of
**the wgpu backend's Vulkan HAL on this class of hardware**, not of the
hardware or of Vulkan itself.

Direct measurement (`crates/gpu-core/tests/vram_overhead.rs`, `nvidia-smi`
deltas around known allocations):

* Allocation alone (`gpu.storage(n)`, never written): **1.00x**, every time.
* Allocation + any upload (`write_f32`/`storage_init`, any `BufUsage`
  combination, `COPY_SRC` present or absent): **exactly 2.00x**, every time.
* Splitting the same upload into 64 MiB chunks via `write_at`/
  `write_f32_chunked` (added to `backend_api::Backend` and all three native
  backends for this investigation): still **2.00x**. So it is not "wgpu's
  staging belt sized to the biggest single call" - the resident cost tracks
  *cumulative bytes ever written to that buffer*, not any call's size, and
  chunking cannot bound it.
* The same probe on **brain's own native Vulkan backend**
  (`crates/backend-vulkan`, ash + naga, selected via `--device vulkan` /
  `Gpu::try_new_vulkan`) instead of the default wgpu backend: **1.00x**. Its
  `with_staging` (`crates/vulkan/src/context.rs`) reuses one shared, bounded
  staging buffer across every upload; wgpu-core evidently keeps a per-upload
  (or per-buffer) shadow instead and never frees it.

**The fix is `--device vulkan`, not a wgpu-level change**, and it is
model-agnostic: every model that quantizes to int8 for capacity reasons
(`crates/qwen3/src/q8.rs`, `zimage`, `flux2`) gets roughly double the effective
VRAM budget for free by preferring the native Vulkan backend over the wgpu
default on this class of hardware. It does **not** relax the "fp32 arithmetic
only" invariant or change any kernel - it is purely a device/backend
selection at residency-placement time. The native Vulkan backend's own
coop-matrix (tensor-core) kernel is unavailable without `glslc`/
`glslangValidator` on `$PATH` (falls back to the scalar kernel), so this trade
needs a real throughput measurement per model before it is made a *default*;
today it is a documented, available override, exercised by one model's
residency plan to fit int8 weights across two GPUs without a CPU-expert
fallback.

**Do not re-derive "2x on non-ReBAR" as a hardware fact in a new model's
docs** - cite this lesson and, if it matters for that model's memory budget,
measure with `vram_overhead.rs` on the backend that model actually plans to
run.
