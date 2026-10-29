<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 374. A page-locked free on the decode lane waits for the whole device, and a prefill round recorded in one call holds the scheduler for the launch queue's drain

Question: with background prefill on, decode steps run on the urgent stream and prompts on the normal one, yet on a
GH200 (Qwen3.8-27B Q8_0, `perf run sweep --workload chat --ladder 1,4,16`) the inter-token gap p99 was 202 ms at
concurrency 4 and 357 ms at 16, max 393 and 410 ms: about one 1024-token prefill round (0.3 s). Where does a decode step
wait for a prompt?

Not where the setup suggested. Per-iteration host timestamps of the scheduler's phases (temporary) and `gdb` breakpoints
on the driver's blocking entry points showed two stalls, both on the scheduler thread, neither a device-side fence:

1. **`start_running` (the first token of a prompt that has just been read in) took 145 ms.** The urgent head step spent
   all of it in `GraphCache::plan`: a graph-named buffer had been freed (the allocator epoch moved), so the stale
   graphs were dropped, and dropping a `LiveGraph` called `cuMemFreeHost` on its 96-byte staging (143 ms in one
   call, 0.0 ms once the device was idle). `cuMemFreeHost` and `cuMemFree` wait for every stream of the context, so the
   urgent handle waited for the normal stream's whole queue (the next prompt's round, queued a moment before). No fence
   was involved: `fence_in` waits were `pending=false` and skipping them changed nothing.
2. **`pump_prefill` with `wait = false` took 147 ms for a 960-row round.** `run_prefill_chunk_stage` recorded all of the
   round's 1425 launches in one call; the stream's launch queue holds about a thousand, so the host blocked in
   `cuLaunchKernel` (100 of the 141 ms was the launch loop at ~70 us a launch against ~5 us unblocked) until the device
   had drained enough of the round to take the rest. A short round that runs as decode steps (24 rows, six groups) did
   the same with a blocking read per group (112 ms). No decode step can be issued meanwhile.

Fix: (1) `GraphCache::retire` keeps a retired graph's page-locked staging and device parameter block for the next capture
(`spare_params`, at most 16, reused when not more than twice as large) instead of freeing them, so retiring a graph
issues no driver call that waits for the device. (2) A chunk round is `begin_prefill_chunk` + `advance_prefill_chunk`:
the layer loop (`StackState`) and a short round's groups (`ExactRound`) resume where they stopped, and the background
pump records for at most `PUMP_RECORD_BUDGET` (16 ms) per call, at least one layer or group, then returns to its decode
step. Results are bit-identical (`a_chunk_round_recorded_layer_by_layer_equals_the_whole_round`,
`a_short_round_issued_a_group_at_a_time_equals_the_whole_round`).

Measured (same sweep command, seed 1, gpu1): inter-token gap p99/max 44/57, 202/393, 357/410 ms became 44/59, 120/201,
179/198 ms at concurrency 1/4/16; output 65.5/130.9/164.6 became 64.0/136.4/182.8 tok/s; TTFT p99 447/1965/10023 ms
became 531/2032/8973 ms (concurrency 16 queues; the others within run-to-run noise).

Rule: on a lane that must not stall, grep the path for calls that wait for the whole device before blaming a fence:
`cuMemFree`, `cuMemFreeHost`, a read, a `poll_wait`. A breakpoint on those entry
points with a 6-frame backtrace found this in one run; `fence_in` logging with `cuEventQuery` showed the fences were
already complete. Host work that is paced by a bounded device queue (record a long round, enqueue a long copy) must be
sliced by host time, not run to completion, wherever a latency-bound loop shares the thread.

Not done, still visible: the MTP head's prompt feed (`mtp_after_prefill` in `finish_slice`, 42-56 ms of synchronous
normal-lane work per prompt); decode steps run 1.5-2x slower while a prefill shares the card (device-side SM
sharing, which stream priority does not change); `dead uniform` blocks are still freed by `cuMemFree` in `poll_wait`
on any handle that is not holding freed blocks. `a_lone_prompt_reads_in_at_least_as_fast_in_the_background` and
`a_generating_sequence_keeps_generating_while_a_long_prompt_is_read_in` (`gguf_resident_real`) fail before and after
this change (1.56 s against 0.76 s sliced for a lone 1731-token prompt; longest decode gap 0.81 s before, 0.17 s after,
against a 0.125 s bound).
