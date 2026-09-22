<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 90. A `Step` is only meaningful to the `Gpu` handle that created it - an unused helper function is not evidence the bug it fixes was already fixed

`crates/pulid/src/model.rs` defines `joint_kernels()` - one kernel list
covering both FLUX.1's dispatch indices and PuLID's, built specifically so a
`PulidCa`'s injected `Step`s resolve against the SAME pipeline index space the
DiT's own `Step`s do (`flux1::inject`'s module doc states the contract
outright: "the steps an implementor pushes MUST be built from the SAME
`gpu_core::Gpu` handle the model was built with"). The only caller of
`joint_kernels()` in the whole workspace was a test
(`crates/pulid/tests/parity.rs`). Production code - `pulid::caps::Bundle::load`
- built the DiT from `Flux1::load`'s own internal `Gpu::new(flux1::KERNELS)`
and built `PulidCa` on a *different* `Gpu::new_like(crate::model::KERNELS)` -
two independent kernel lists, hence two independent index spaces, feeding one
dispatch list. Every existing unit/manifest test for PuLID passed regardless,
because none of them ever ran a real conditioned forward (no reference
checkpoint has existed in this workspace for `flux1`/`pulid` until this
change's own script provisioned one) - the bug was invisible to every gate
that could run without real weights, and the ONE artifact that would have
caught it (a correctly-wired test) was sitting unused in the same crate.

**The thing to carry forward**: when a crate defines a
correctness-contract-shaped helper (a "build these two things from the same
handle" function, a "validate this invariant" assertion) and grep shows its
only caller is a test, that is not proof the contract is honored elsewhere -
it is a specific, checkable claim ("production code does NOT call the thing
that makes this correct") worth verifying by name, the same way a claimed
kernel property gets checked against the kernel's own source rather than
trusted from a report (lesson-adjacent to the "hypothesis until checked"
rule elsewhere in this file). An unused correctness helper is a live defect
wearing the shape of a fix.
