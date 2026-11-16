<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 480. A kernel tuning sweep charged its synthetic problems to the model it tuned, so a cold start reported ten times the memory of a warm one

**Lesson:** A charge against the memory ceiling lives as long as its `Gpu` handle, not its buffer. Work that churns
temporaries on a long-lived handle releases their charges in a scope (`gpu_core::with_transient_charges`) or runs on a
throwaway handle (`gdn_prefill_tune`), never on the model's handle as it is. A test that measures charged bytes is
only deterministic once it decides whether its start is cold (no stored tuning table) or warm.

## What happened

`qwen3`'s `mem_budget_inference::inference_only_build_stays_under_a_generous_device_budget` failed on CUDA with
15.9 GiB, then 26.7 GiB, charged for a Qwen3-0.6B inference build of 2.6 GiB, and passed on every rerun; a bisect over
440 commits found every commit good. The build itself never changed: `Gpu::wrap_on` measures a device's native kernel
choices (`native_tune::ensure`) on the new handle the first time a build of brain meets the card, and the sweep builds
gigabytes of synthetic weight and activation problems on that handle. Each buffer is dropped when the sweep returns,
but its grant stayed in the handle's `grants` for the handle's lifetime, so the model carried the whole sweep as its
own charge. Only a start without the stored table (`native-tune-*.txt` in the cache directory) measured, so the
failure followed the cache, not the code: the first run after a build or a cache wipe failed, every later one passed.

Under a real ceiling (`--limit-vram-total`) the same cold start was refused memory it no longer held.

## What changed

- `Gpu::hold` keeps a grant to the end of the enclosing `with_transient_charges` scope on that thread when one is
  open, else for the handle's lifetime; `native_tune::ensure` runs its sweep inside one. The charges still count while
  the sweep runs, so a sweep cannot pass the ceiling either.
- `gpu-core` `memory_limit::a_measured_kernel_choice_leaves_no_charge_on_its_handle` builds a tuned handle on a fresh
  cache directory under a ceiling and requires nothing charged on the handle or the ledger afterwards.
