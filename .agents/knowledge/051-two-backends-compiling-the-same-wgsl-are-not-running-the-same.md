<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 51. Two backends compiling the same WGSL are not running the same code, and only a per-backend roofline can see it

`backend-vulkan` and `backend-wgpu` both feed `crates/kernels`' WGSL through
naga. That made "the kernels are identical, so kernel efficiency must be
identical" feel like a safe assumption, and it was wrong by a factor of two:
the native Vulkan path used `naga::back::spv::Options::default()`
(`force_loop_bounding: true`, bounds policy `Restrict`) while the wgpu path
asks for `ShaderRuntimeChecks::unchecked()`, and `backend-cpu` asks Cranelift
for `MemFlags::trusted()`. Same source, three backends, two rule sets - and
the odd one out measured 5.05 TFLOP/s where its sibling measured 10.62 on the
same card.

Three things this cost, each generalisable:

1. **Nothing had ever rooflined that backend.** `gpu_core::roof` existed and
   worked; every caller just happened to run on the default. A backend with no
   measured roof has no measured anything - the deficit was invisible for as
   long as it went unprobed, and one `roof::measure` call per backend found it.
2. **The roof memo keyed on the adapter description alone**, so once both
   backends were probed in one process the first one's number was served as
   the second's. A roof is a property of the (device, BACKEND) pair. Any cache
   whose key omits a dimension the value actually depends on will eventually
   answer confidently for the wrong thing - and here the wrong thing was off
   by 2x, which is large enough to invert a design decision.
3. **A cross-backend switch has to be spelled once.** The fix reads the same
   `BRAIN_GPU_CHECKED` variable the wgpu backend reads, because a second,
   independently-named knob is how the two drifted apart in the first place.

Corollary for anything comparing backends: measure the roof on the backend you
are about to grade against it, never the roof the other backend left in the
cache; and when a "% of peak" moves after a backend swap, suspect the
compilation options before the silicon.
