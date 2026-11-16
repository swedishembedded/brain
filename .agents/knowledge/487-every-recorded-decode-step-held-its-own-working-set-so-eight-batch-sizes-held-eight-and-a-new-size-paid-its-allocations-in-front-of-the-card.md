<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 487. Every recorded decode step held its own working set, so eight batch sizes held eight, and a new size paid its allocations in front of the card

**Lesson:** A cache of tapes is a cache of working sets: a recording keeps every temporary it allocated, and an
engine records one per batch size. The steps never run at once, so they can bind one set, the widest step's
(`gpu_core::tape::TapeScratch`, `Gpu::record_tape_on`), and the plan then charges exactly that set beside a prefill
round.

## What happened

The released 35B-A3B GGUF planned 64 sequences of 75520 tokens on one GH200, and `perf run serve` at c=64 x 8K
never ran them: refused allocations (a 288 MiB prefill-round buffer) backed the scheduler off, in one run to 14
sequences at once, in another to one, with 60 of 64 requests refused. Measured on the real checkpoint
(`live_tally` device bytes): a recorded step holds 0.02 / 0.21 / 0.78 GiB at 1 / 16 / 64 rows, and the eight kept
tapes of a batch shrinking from 64 to 57 held 5.6 GiB. The plan charged the larger of `serve_peak_bytes(64)` (3.1 GiB,
which also misjudges the step: 0.02 GiB at 16 rows) and one prefill round, not both, and not the tapes.

The same recordings were most of the serving idle: at c=16 the batch changes size with every admission, each new
size was recorded with fresh zeroed allocations, and the card idled 20 % of the window (4.8 s in gaps over 1 ms,
most between buffer clears). A held recurrent-state slot for the prompt (the suspected cause) was measured too: no
change, its blocks come back from the engine's retained free blocks without a driver call.

## What to do

- Record against one `TapeScratch` per cache of recordings that never run concurrently and whose outputs are read
  before the next replay: `storage` calls during the recording take slots in call order (a slot at least the request;
  a larger request grows it and the cache drops the tapes recorded before, `TapeScratch::replacements`). A host write
  into a slot while recording is refused (inputs a replay rewrites are allocated before it opens).
- Charge the widest step's set beside a prefill round (`qwen35moe::serve::decode_step_bytes`, calibrated against the
  shared set: 0.021 / 0.218 / 0.892 GiB at 1 / 16 / 64 rows).
- Keep separate sets for recordings whose outputs outlive the next replay of another (the dense engine's verify
  rounds, whose hidden block the drafter reads, have their own).

## Evidence

`gpu-core/tests/it/tape.rs` (`recordings_against_one_scratch_share_the_widest_working_set`,
`a_host_write_into_shared_scratch_is_refused`); `qwen35moe/tests/it/decode_tapes.rs`
(`recorded_steps_of_every_batch_size_hold_one_working_set`). On gpu3: the eight tapes of 64..57 rows 5.6 -> 0.89 GiB;
c=64 x 8K, 64 requests: peak sequences at once 29 -> 64, 4 -> 64 completed, 1.23 req/s; c=16 busy 79.5 -> 96.8 %.
Ledger: `.agents/roadmap/qwen35moe.md`.
