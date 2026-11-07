<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 25. "Batched prefill" that batches the readback, not the dispatch is not batched

**Lesson:** "Batched prefill" that batches the readback, not the dispatch, is not batched.
Count device ops; wall-clock cannot tell "faster" from "O(1)".

## What happened
`Qwen::prefill`'s fix replaced a per-token `decode_submit` loop with one call to the batched
primitive.

An earlier attempt at "fixing" prefill performance batched only the readback (one `map`/fence at
the end) while still issuing one GPU submit per token underneath. It measured faster than the
naive `step()`-per-token loop (fewer host↔device round trips), so it read as progress. But it was
still `O(T)` submits for a `T`-token prompt: the same defect class in a lighter disguise.

## Why
"Faster than before" and "actually O(1) [per chunk]" are different claims, and a wall-clock-only
benchmark cannot tell them apart.

## What to do
Gate on a device-op COUNT (`gpu_core::DeviceStats.submits`), which is insensitive to how fast any
individual submit happens to run on the current machine. The gate this lesson names,
`prefill_submits_scale_with_chunks_not_with_token_count`, is what survives.
