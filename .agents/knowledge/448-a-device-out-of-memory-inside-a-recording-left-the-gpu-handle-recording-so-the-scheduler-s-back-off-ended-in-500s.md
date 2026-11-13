<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 448. A device out-of-memory inside a recording left the GPU handle recording, so the scheduler's back-off ended in 500s

**Lesson:** An unwind that passes through `Gpu::begin_tape` .. `end_tape` leaves the handle recording for the rest of the
process: every later `read`, `poll_wait` or recording asserts. The scheduler's typed back-off on a device out-of-memory
(#445) is only as good as the state it unwinds out of; recordings are now scoped by a guard (`Gpu::record_tape`).

## What happened

Qwen3.8-27B-FP8 on one GH200, `brain serve --max-batch 64 --ctx 1024` (default `--reserve-gb 2`), 64 concurrent requests of
400 tokens (2026-11-13). The first load of 64 short requests passed. The second logged the scheduler's back-off (`the device
ran out of memory (1510848 bytes refused); 61 sequence(s) go back to the queue and at most 30 run at once`) and then 61 of
the 64 requests failed with `panicked while running: Gpu::poll_wait while a tape is being recorded: nothing recorded has been
submitted`, and kept failing: the allocation that was refused was made while a decode step was being recorded as a tape, the
unwind skipped `end_tape`, and `taping` stayed set on the handle. This is the mechanism behind "every request then 500s and the
model is unservable" of #445; a refused graph launch (`cuGraphLaunch`) raised an assertion rather than the allocator's typed
payload, the second way to the same end (now typed: `refuse_after_launch_failure`).

## Fix and evidence

- `Gpu::record_tape()` returns a guard; `TapeRecording::finish()` ends the recording, and dropping the guard any other way
  abandons it. The recordings of the Qwen3.8, Qwen3.5-MoE and DeepSeek2 decode steps and the MTP draft plans use it. Gated:
  `crates/gpu-core/tests/it/tape.rs` (a recording an unwind passes through leaves the handle usable).
- A graph launch that runs out of memory after its first graph unwinds with `DeviceOutOfMemory` (`bytes` 0: the driver does not
  say) instead of asserting; the first graph failing still falls back to step-by-step submission. Gated on the CPU:
  `backend-cuda` `a_launch_that_runs_out_of_memory_after_the_first_graph_unwinds_with_the_typed_refusal`.
- Real weights, same load after the fix: 64 requests of 400 tokens, three loads in a row, 64 of 64 answered each time; the
  scheduler halved its batch through 29, 14, 7, 3 to 1 rows on five OOM events and recovered (the third load took 14 s).
- The reserve the batch needs: `--reserve-gb 2` (default): back-off to one row, first loads 20 s; `--reserve-gb 6`: five OOM
  events, 15 s; `--reserve-gb 12`: none, 8 s. `brain serve` therefore raises the default reserve by 1 GB per 4 rows of
  `--max-batch` above 16 (64 rows: 14 GB) and says so at startup; an explicit `--reserve-gb` is never changed.
- `ChatFineTune::run_with` returns `Error::DeviceOutOfMemory` (and `brain/train` fails with the `unavailable` class) instead of
  unwinding, so a full card during a training run is an error a caller acts on.

## Not done

A configuration that cannot fit at all (Qwen3-4B's default plan in a 13 GB share of a card, reported by a capture run and not reproduced here: HTTP 400 and
500, the fix being `--ctx 16384 --kv-cache fp8`) still fails without a message naming the setting to change; open in
`.agents/roadmap/serving-runtime.md`.
