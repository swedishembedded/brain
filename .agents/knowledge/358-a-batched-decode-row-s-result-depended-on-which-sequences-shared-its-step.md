<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 358. A batched decode row's result depended on which sequences shared its step

Symptom: a served 27B Q8_0 GGUF (ctx 32768, int8 KV, `--max-batch 8`, greedy,
fixed seed) given one identical task-derived request five times sequentially
returned bit-identical completion token ids AND per-token log-probabilities.
Six or eight CONCURRENT identical requests returned two distinct token
sequences with the first difference at token 7; two concurrent requests were
identical. The model had no randomness in its sampling: the difference was in
the arithmetic, and it was a function of the co-batch.

Three mechanisms, all row-count or batch-size gates on a path that shares one
dispatch across unrelated sequences:

1. **The int8 linears switch kernel with the row count.**
   `backend_api::select::i8_gemv_max_rows` and `DECODE_REGIME_MAX_ROWS` route
   `Ops::matmul` to the packed GEMV for a decode-sized `m`, and inside
   `kernels-cuda/cu/matmul_i8_gemv.cu` the same launch switches to its int8
   tensor-core block from `I8_GEMV_MMA_MIN_ROWS` (5) rows. That block applies
   each 32-product group's scale once with a fused multiply-add where the
   reference folds 64 virtual accumulators, so it agrees to f32 rounding of
   the running sum, not bit for bit - the kernel's own documented contract
   ("bit-identical below 5 rows, int8 tensor cores from 5 rows"). A batched
   decode sent all `bsz` rows through one `Ops::matmul`, so rows 1-4 and rows
   5-8 of the same batch ran different arithmetic than a single request did.
2. **The int8 head follows the same gate.** `stream::head_rows_on` builds a
   `[rows, vocab]` activation and one `Ops::matmul` at `m = bsz`, so the
   logits themselves switched kernel at `bsz = 5`: the sampled token could
   change even if the body had been exact.
3. **The fused decode's split count is sized from the batch and from the
   batch-wide `cap`.** `kv_tier::flash_splits` aims the dispatch at a
   workgroup target divided by `batch * n_kv_heads`, and `cap` in
   `run_decode_batch` is the LONGEST live sequence rounded up to a bucket. Both
   change a row's online-softmax tile boundaries when a neighbour arrives or
   leaves, even though the neighbour's own keys are never read.

Fix: `--exact-decode` (model-agnostic serving policy switch, default off).
On, `Qwen35::run_decode_batch` runs the batch in groups of
`select::exact_decode_group_rows` (one fewer than the tensor-core block's
first row count where the device has it, the GEMV's whole row range
otherwise), each group a complete step on the bit-identical kernel; the GGUF
resident's head projects the same groups; and every row's attention uses the
configured per-sequence context as `cap` and the batch-1 split count
(`FlashDecodeShape::pin_splits`), with the block-table width pinned to the
context too so it cannot follow the longest neighbour. Over-splitting a short
sequence is one degenerate partial record, which is the design's own cheap
case. Speculation is turned off in this regime: a verify pass is a sequence's
consecutive tokens and is not grouped, so claiming the regime for it would be
false.

Cost, reasoned from a measurement already in the tree rather than assumed: the
batched int8 GEMV shares one read of every weight across the rows, so a group
of 4 at `bsz` 8 is two passes where the step made one. They are not the same
rate, either: `crates/gpu-core/tests/i8_gemv_batched_variants.rs` measured the
8-row tensor-core block at 1.48-1.99 TB/s of weights against the dp4a block's
1.0-1.26 TB/s at the 27B projection shapes. Two dp4a passes against one
tensor-core pass therefore cost about 2.4-3x the projection weight time, so a
theoretical batched step that streams 28 GiB once goes to 56 GiB at the slower
rate: batch-8 aggregate throughput falls to about the batch-4 level (~half of
today's), and the int8 head follows the same gate. The group size is the widest
that is still bit-identical, so this is the minimum cost of the property on
this kernel; a group of one would be eight passes.

Gates: `exact_decode_groups_make_every_batch_row_bit_identical_to_the_same_row_alone`
(`crates/gpu-core/tests/i8_gemv_native.rs`, CUDA; the op-level contract at
`m = 1`, the switch point 5 and 8), `pinned_splits_are_the_single_sequence_splits_at_every_batch`
(`crates/model/src/kv_tier.rs`), and the CLI acceptance in
`crates/cli/tests/serve_cli.rs`. The end-to-end claim (one request's tokens and
per-token log-probabilities unchanged while co-batched) needs the real 27B on
a GPU and is left to the serving run.
