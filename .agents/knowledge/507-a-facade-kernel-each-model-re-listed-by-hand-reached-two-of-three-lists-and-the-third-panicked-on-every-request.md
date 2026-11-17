<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 507. A façade kernel each model re-listed by hand reached two of three lists, and the third panicked on every request

**Lesson:** a set every consumer must carry whole is appended whole from its one owner, never re-listed entry by
entry. Each consumer also needs a test that asserts it carries the set, and a gate that finds the consumer that has
no such test. A required-name check that runs only where an engine is built lazily runs on a live request, not in
`cargo test`.

## What happened

A commit added `embed_i8` to `model::ops::REQUIRED_KERNELS` (`kname::ALL`) and to the canonical
`model::ops::kernel_list()`. It registered the kernel by hand in the qwen3 and qwen35moe pipeline lists. The qwen35
list, which re-listed every façade kernel the same way, did not get it. Every test passed. Serving
Qwen/Qwen3.8-27B-FP8 then panicked on every request: `qwen35: Ops::new: kernel 'embed_i8' is not registered on this
Gpu`. The residency pool builds the engine at a model's first request, so nothing reached `Ops::new` on the qwen35
list before production did.

The same drift had happened before: `qwen3::serve`'s hand-kept list ran 15 kernels short. The fix then was the
canonical `kernel_list()` and `assert_kernel_list_complete`. But three model lists kept their hand copies to register
`matmul_reg2` over the `matmul_reg3` source, and only qwen3 tested its list.

## Why

- `Gpu::kernel_index` resolves a name to its FIRST slot. Appending the canonical list after a model's own entries
  would have kept the twin. The lists were copied by hand because no shared helper with first-wins dedup existed
  to make that safe.
- The completeness assertion existed but was opt-in per crate. Nothing found a crate that built an `Ops` without
  calling it.

## What to do

- Compose every pipeline list an `Ops` is built on with `model::ops::extend_kernel_list`. List the model's own
  kernels first (fixed slots, registered twins), then append `kernel_list()` or `training_kernel_list()` whole, then
  the other shared sets (`cq_kernels`, `model::saved::kernels`, ...). The helper dedups by name, and the first
  registration wins.
- Test each such list with `model::ops::assert_kernel_list_complete`, or `assert_training_kernel_list_complete` for
  a list an `Ops` trains through: that one also requires `frozen_base_kernels`, which only resolve at first
  dispatch.
- `make check/ops-kernel-lists` fails a crate that builds an `Ops` in `src/` and calls neither assertion. Exceptions
  go in `scripts/gates/ops-kernel-lists.allow` with a reason.

## Evidence

The qwen35 list's completeness test fails on the old list (`missing 1 kernel(s) ...: ["embed_i8"]`). A dump of every
qwen3/qwen35/qwen35moe pipeline list before and after the change shows the same name-to-source map. The only
difference is `embed_i8` added to qwen35's lists. Every changed position comes after the hand-numbered static slots.
