<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 409. Verifying speculated rows over a bf16 state pool is exact only if the state is rounded after every row, and running the plain decode kernel over a row loop also removes the staging copies

**Lesson:** A kernel that replays a recurrence for verification must replay it with the plain
kernel's rounding points, so the state has to be rounded after every row. Reusing the plain decode
kernel over a row loop makes the replay exact and also removes the staging copies.

## What happened
Question: greedy speculation over a bf16 recurrent-state pool was not token-identical to plain
greedy decoding (the test had been relaxed to a 12-token agreement), and 5.6 ms of a 23.6 ms
four-sequence verify round was staging copies. One cause?

## Why
Plain bf16 decode (`brain_gdn_decode_pool_stream_bf16`) keeps a head's state tile in shared memory,
rounds it to bf16 in place and writes the bf16 tile and the conv window back after EVERY token.

The first bf16 verify gathered the pool rows into f32 staging, ran the f32 verify kernels over all
of a sequence's rows and rounded once after the last row (`verify_through_staging`). Intermediate
rows were computed from an unrounded state, so a verified row differed from the plain step it stands
for.

The copies (gather and scatter of the whole pool rows, 2.8 + 1.9 + 0.9 ms a round, nsys, 8K x 4,
`--speculative mtp`) were pure overhead.

## What to do
`brain_gdn_verify_pool_stream_bf16` is the plain streaming body with a row loop
(`brain_gdn_stream_body<VERIFY>`; plain decode is the one-row case of the same function).

- The state tiles stay resident in shared memory across a sequence's `counts[bi]` rows, so each row
  reads the tile the row before rounded, and the conv window is read back from the bf16 pool row the
  thread wrote.
- After every row but the last the tile and window are written to the f32 snapshots as the widening
  of the bf16 values.
- A restore (`brain_gdn_commit_pool_bf16`: f32 snapshot -> bf16 pool row, a narrowing that is exact
  for such values) needs no staging either.
- The staging path, `VerifyLayout::ident` and `gdn_commit_pooled_bf16` are gone.

Rule: a kernel that replays a recurrence for verification must replay it with the plain kernel's
rounding points; widening the storage precision for the replay is not a neutral implementation
detail, and a copy to a staging layout to reuse a kernel is a cost to measure before accepting.

## Evidence
Measured on gpu1:

- `crates/model/tests/it/gdn_verify_native.rs`: every verified row (layer output), the pool after
  the last row, every snapshot and the bf16 commit are bit-identical to repeated plain bf16 decode
  steps, at 1..16 key heads, 1..3 value heads a key head, 1..8 rows, ragged batches. Red first (the
  staged path could not run it), green after. compute-sanitizer memcheck and racecheck on that test:
  0 errors.
- `crates/qwen35/tests/hf_fp8_speculative.rs`: greedy speculation over bf16 pools token-identical to
  plain greedy, request for request (the tiny fixture does not discriminate: the strict assertion
  also passes on the staged path, so the kernel test above is the gate that bites).
- nsys, 8K x 4 `--speculative mtp`, GPU time of verify + staging + commit kernels a verify round (48
  linear layers): staged 9.7 ms (141 rounds, 1368 ms) -> 1.57 ms (319 rounds, 501 ms: verify 21 us a
  layer, commit 12 us).
