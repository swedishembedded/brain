<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 535. A per-row CPU fan-out spends a wide pool on wake-ups

**Lesson:** size a CPU parallel loop by the work it carries, not by its row count. The CPU backend's native fast
paths cut their output into one rayon task per row (`par_chunks_mut(row)`) or per `threads * 4`th of the rows. On a
288-thread host a tiny model's softmax over a few dozen keys became hundreds of tasks of a few dozen floats, and every
thread spent its time waking, stealing and parking. The fix is one shared grain (`backend_cpu::par::MIN_TASK_WORK`,
64K work units) that every fast path's loop goes through.

## What happened

2026-11-19, the SDK's `decision_promote` test (two 1500-step trainings of a tiny `decide` encoder head, then
promotion), CPU backend on a 288-core GH200 node:

- **33 minutes (1974 s) for a test whose arithmetic is seconds.** The process used about 165 cores' worth of CPU time per
  second of wall time. `gdb` stack snapshots showed the test thread in `fast_ops::attn_softmax_cross` under
  `Decide::accumulate_batch`, and the pool threads in `rayon` `join_context` over the softmax's per-row
  `ChunksMutProducer`, contending in `Sleep::wake_specific_thread`.
- **The pattern was a class, not one kernel.** The same unsized fan-out was in every row-shaped fast path: the row
  softmaxes (`attn_softmax_cross`, `attn_softmax_causal`), the per-head attention products and their gradients
  (`gqa_*`), the dense and int8 products' row split, `matmul_dx`/`matmul_dw` and their gated MoE twins, `scale_add`,
  the masked cross-entropy pair, `transpose`, `silu`/`silu_mul`, `bn_eval`, `concat2`/`concat_split`/`chan_place`,
  `upsample2`, `gn_apply`/`gn_part`, the conv band loop and depthwise/bias planes, the grouped MoE dispatch and
  `host_gemm::blocked_linear`. Only the NEON GEMM (`FLOPS_PER_TASK`), the JIT dispatch (`MIN_CHUNK_INVOCATIONS`) and
  the zero-fill and GEMV column split (`MIN_TASK_BYTES`, now merged into `par::MIN_TASK_WORK`) had a grain, each its
  own constant.
- **A one-chunk `par_chunks_mut` is not inline.** Called from outside the pool, rayon injects even a single task into
  it and blocks the caller, so a below-grain job must skip rayon entirely to cost nothing.

## What to do

- Loop through `par::items_mut` / `spans_mut` / `for_each_index` with the work of one item (floats touched, or
  multiply-adds for a product). They group whole items into tasks of at least `MIN_TASK_WORK`, at most about four
  per thread, and run a job below the grain on the calling thread without touching the pool.
- Keep spans whole items (whole 8-lane vectors for an elementwise SIMD kernel), so the result is the serial loop's
  bits on any pool size; the backend's `row_kernels_give_the_same_bits_on_any_pool` test holds that.
- `par::rows_mut`, `chunks_mut` and `map` still split by item count, because their callers' per-item cost is unknown
  to them; a caller with small items moves to the grained form with its own work estimate.

## Evidence

`decision_promote` went from 1974 s to 76 s on the CPU backend, 26x (row C1 of
`.agents/roadmap/training-efficiency.md`); `par::tests::small_work_stays_on_one_thread_and_large_work_fans_out` fails
without the grain. After the fix, stack snapshots show the test thread in the JIT dispatch's own fan-out, which already
has its grain (`MIN_CHUNK_INVOCATIONS`).
