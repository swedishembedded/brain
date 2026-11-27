<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 613. A dry KV pool read as the card's limit held a half-empty card at five sequences

**Lesson:** a failure signal must name what failed. A paged KV pool that runs out of free blocks says the scheduler's
block accounting was short; a device allocation that fails says the card is full. Treating both as "the card holds
fewer sequences" lets one accounting slip halve the batch, and a back-off that gives the batch back one sequence per
32 iterations never catches up with the next slip.

## What happened

2026-11-27, brain a2e6c80a9 servers on GH200 cards, an agent runtime's trials (contexts 5-30k tokens, about 50 hot LoRA
adapters on Qwen3-4B, Qwen3-8B and Qwen3.8-27B-FP8 resident together). n547 gpu1, no training: `/health` queue_depth
39, kv_total_tokens 196176, kv_free_tokens 194384, 49 of 97 GB in use; the log repeated
`device out of memory for the paged KV pool: paged KV pool exhausted` and `at most M run at once`, M falling 25, 11,
8, 5; cards ran at about 20% SM, and the per-model admitted width (#610) advertised that narrow width to the routers.

## Why

- Admission charges a prompt only for the blocks it does not adopt from the prefix cache, but the prompt adopts them
  when its prefill starts. Between the two, the next admission (and the decode-room pass) reclaimed cold cache blocks,
  and the cold end of the cache is exactly every conversation's earlier turn: the prefill admitted first found its
  history gone and ran the pool dry.
- `MemoryPressure` read the dry pool (`bytes 0`) like a device failure: the width halved to in-flight/2, and grew
  back by one sequence every 32 calm iterations (5 to 64 takes 1888 iterations), so it never recovered between
  slips. `kv_free_tokens` read nearly total because the requeued sequences gave their blocks back and the reading
  counts cached blocks as claimable.
- Separately, the 4B's pool is 12261 blocks x 16 (196k tokens), sized once at construction from the bytes the card had
  free then; on a whole GH200 share the same plan is a 1M-token pool (84 GiB).

## Fix

- A prefill admitted but not started that counts on cached blocks makes a later admission that would need a reclaim
  wait an iteration; the decode-room and resume reclaims re-read what such prefills adopt
  (`Scheduler::reclaim_for_running`).
- `model::paged::pool_ran_dry` tells a dry pool from a refused allocation; a single dry pool narrows nothing.
- After a device failure the limit grows by half every 4 calm iterations up to just under the width that failed, and
  past it at once when the decoder reports more free device memory than the failure left
  (`PagedDecoder::device_free_bytes`, `cuMemGetInfo` on CUDA); otherwise it probes by an eighth every 32. A drained
  scheduler forgets the limit.

Specs: `crates/model/tests/it/serve_device_oom.rs::after_a_card_runs_out_the_batch_is_whole_again_within_a_few_dozen_iterations_once_it_has_room`
(red before: still 3 of 32 after 48 iterations), `under_a_stream_of_arrivals_the_running_width_tracks_the_room_the_card_has`
(red before: mean 9.8, least 3 of 16 after the room came back), `a_kv_pool_that_ran_dry_once_does_not_narrow_the_batch`,
`crates/model/tests/it/serve_agent_burst.rs` (96 agent trials, 8 turns, a 196k pool: 2-4 dry pools before, none after).

**How to apply:** the harness also shows the pool, not the back-off, bounds an agent burst: the same burst runs a mean
width of 16 on the 196k pool and 55 on a 1M pool, in 7.7k versus 2.2k iterations. Size a serving pool to the card's
free memory, not to what was free at startup.
