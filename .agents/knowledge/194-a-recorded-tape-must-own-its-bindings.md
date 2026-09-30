<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 194. A recorded tape must own its bindings, or the next recording rewrites them

Models record their training forward and backward once and resubmit the same
`Step`s every step. On wgpu a step owns its bind group, so that is safe. The
Vulkan backend recycled a transient step's uniform buffer and descriptor set
into its free pools as soon as the batch that ran it retired. The step itself
was still held by the model. The next dispatch built anywhere on that handle
popped the "free" set and rewrote it with its own buffers and params. The
held step then ran with the newcomer's bindings on its next submit.

Nothing failed loudly. On a Qwen3 training build, one decode step between two
forwards moved the recorded forward's loss from `3.1818075` to `3.1793568`,
while the CPU backend gave `3.184268` for the same sequence (the decode also
overwrote the batch's token buffer, which is expected). The difference came
only from recycled bindings. With an adapter attached and detached in
between, the next resubmitted training step never completed, and the
`BRAIN_GPU_WAIT_S` timeout fired. A pure training loop never builds a step
between resubmits, so every training gate passed. The step cache
(`Gpu::enable_step_cache`) resubmits held steps by design and was exposed to
the same recycling.

The fix ties recycling to lifetime. Each transient `VkStep` carries an
`Arc<StepLease>` holding its uniform and set. They return to the handle's
`Recycler` only when the last clone drops: the caller's clone and the clones
that pending and in-flight batches keep until they retire. The regression
gate is `backend-vulkan/tests/perf_contract.rs::a_held_step_keeps_its_bindings_across_recycling`.
It was red before the change (the held step wrote nothing into its own
output) and is green after.

The fix exposed a second cost. A step-cache entry now keeps its resources
for as long as it lives. The seam of a decode tape (every dispatch whose
params carry the position) used to leave one never-reused entry per token
until the 65536-entry cap. Each such entry now allocates a fresh uniform. So
`StepCache::put` replaces a call site's previous entry when it was never
reused (`gpu-core/tests/step_cache.rs::a_seam_that_never_repeats_keeps_one_entry_per_site`).

Rule: a backend object that a `Step` names must live exactly as long as the
`Step`, never until an event the step's holder does not control. Test a new
recycling scheme by keeping a step, building and flushing other work, and
submitting the kept step again.
