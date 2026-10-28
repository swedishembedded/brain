<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 371. A pool copy that binds the pool as source and destination is a GPU-only validation error

Question: why did serving the Qwen3.8 27B on the wgpu backend fail with "Attempted to use Buffer with '' label with
conflicting usages ... STORAGE_READ_ONLY and STORAGE_READ_WRITE", when every CPU test passed?

Finding (a defect): `KvPlane::copy_rows` (the prefix-cache and copy-on-write seeding of a KV window) and
`Qwen35::slot_copy` (pooled Gated-DeltaNet state rows) copied rows inside one pool by dispatching `splice` or
`region_copy` with the pool bound as both source and destination, each at its own sub-range. wgpu tracks a buffer's usage
over the whole dispatch, so the `read_write` destination conflicts with the `read` source whatever ranges they cover. The
CPU backend has no usage scopes, so no test saw it; the one existing guard (`assert_no_output_alias`) looked only at the
last binding and sliced dispatches skipped it. The fix is a `copy_within` kernel with a single `read_write` binding.

The guard is now exact: `KernelGrid::written_bindings` is read from each kernel's own source and `Gpu::step`,
`Gpu::dispatch` and the sliced forms refuse any dispatch that binds one buffer to a `read_write` slot and to another slot,
naming the kernel, on every backend. The same class fixed earlier (the anchor bank's `copy_row`) was found by a GPU run.

Rule: a copy between two ranges of one buffer is one binding. Two read-only bindings of a buffer stay legal.
