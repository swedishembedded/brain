<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 517. A backend without memory accounting reports `None`, and every planner reading it treats the card as unmeasured

**Lesson:** `Backend::memory_usage` defaults to `None`, which callers fold into zero or into "not measured" without
saying so. A backend that holds buffers must count them; the counting is one small helper, not a per-backend struct.

## What happened

The CPU, wgpu and CUDA backends each reported live and peak bytes (CPU and wgpu with private copies of the same two
atomics); the Vulkan backend reported `None`. Residency planning, leak checks and cost pricing on a Vulkan device read
no figure, and a model measured on it looked free.

## What to do

`backend_api::MemoryTally` is the one accounting type: `add` at buffer creation and growth, `sub` at drop,
`usage(held_back)` for the report (`reserved` is `live` plus what dropped buffers still hold: wgpu's pending reclaim,
Vulkan's buried buffers), `reset_peak`. CPU, wgpu and Vulkan use it; a handle from `share` counts into its parent's
tally and one from `new_like` starts its own. CUDA keeps its tally in `live.rs`, which also counts non-buffer objects.
The counting rules are tested without a device in `backend-api`; `crates/backend-vulkan/tests/it/memory_usage.rs`
checks the buffers feed it, on any Vulkan adapter, lavapipe included (none is installed on the shared node, so the
gates skip there).
