<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 207. `cuMemFree` is a device synchronisation, and a fast kernel hides behind it

Moving the Qwen3.8-27B INT8 prefill GEMMs onto tensor cores took them from 10
to ~225 TOPS and moved a 256-row round from 1.69 s to 0.55 s. The round was
then still 2-3x slower than its own kernel table said it should be: 333 ms of
device time, 554 ms of wall time.

The kernel table (events around each launch) could not see the difference,
because the missing time was not on the device. Sampling the host with `gdb`
(`thread 1`, `bt`, a few dozen times) put every sample inside `cuMemFree`:
freeing a device block waits for the card to drain, so a layer that drops a
dozen multi-megabyte activations puts the host in lock step with the device
instead of ahead of it - and the same dozen blocks are allocated again by the
next layer. Allocation churn is invisible to a per-kernel profile and to every
counter that measures work.

What it taught, in the order it should be applied:

1. **Before optimising the next kernel, ask where the host is.** A table that
   sums to less than the wall time is a statement about the host, not about the
   kernels. `gdb -p PID -batch -ex "thread 1" -ex "bt 40"` in a loop is enough;
   no profiler is needed to find a free that blocks.
2. **Sample the right phase.** A profile tool that times kernels serialises
   them (an event and a wait per launch), so sampling it during the timed
   region shows the timer, not the workload. `BRAIN_PREFILL_PROFILE_SKIP_TABLE=1`
   stops after the production rounds for exactly that.
3. **Small blocks are not free either.** The first fix held only large blocks;
   the 1-64 KiB per-layer planes were then the ones still reaching `cuMemFree`.
   Only the sub-kilobyte uniforms free without waiting - and a round frees two
   thousand of those, at several microseconds a driver call each way.
4. **A cache must not break the leak contract.** Holding freed blocks on a live
   context made `tests/leaks.rs` fail (it frees on a live context and expects
   the counters back at once). The cache is opt-in per handle for the duration
   of a repeated pass (`Backend::hold_freed_blocks`, switched on by
   `Gpu::scratch_scope`), bounded, and returned when the handle goes; every
   other owner frees exactly as before. And a model whose passes use more than
   one handle (a `share`d `Ops` handle allocates the quantised activations)
   must switch it on for each: the first version enabled it on the model's own
   handle and left the activations - the large blocks - going to the driver.
5. **`cuMemFree` was also the only thing ordering a reused address against
   other streams.** Because it drains the whole device, nothing was ever
   pending on a block that came back from the allocator. A held block is
   reissued with other handles' kernels possibly still queued on it, so the
   cache carries the block's fence and the allocating stream waits on it before
   zeroing or writing (`a_reissued_block_waits_for_the_other_streams_work_on_it`
   queues 400 launches from a second handle and fails without the wait). The
   symptom that exposed it was not a crash: the real-dims chunked-prefill gate
   drifted from 1.00e-3 to 1.95e-3, inside its own bar. A gate whose bar leaves
   room for a defect is a gate that reports it as a number to read, so read the
   number whenever a change that should not move it does.
6. **A shared card makes every number a distribution.** Other processes
   time-slice the card, which only ever adds time to a measurement. Report the
   fastest of several rounds, check the card is idle before and during, and
   compare against a "before" taken with the same binary and the new paths
   switched off, not against a number from another day.
