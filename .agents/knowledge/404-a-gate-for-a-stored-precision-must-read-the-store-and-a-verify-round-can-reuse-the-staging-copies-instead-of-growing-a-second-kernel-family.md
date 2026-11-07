<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 404. A gate for a stored precision must read the store, and a verify round can reuse the staging copies instead of growing a second kernel family

**Lesson:** A quality gate for a stored precision has to read the stored values, not a working copy,
and it must compare teacher-forced scores to tell a numeric near-tie from lost information. A verify
round can reuse the existing staging copies rather than growing a second kernel family.

## What happened
Question: can the recurrent-state pool be bf16 by default (knowledge 396 measured the speed, the
gate was open), and can speculation run over it?

`qwen35_fp8_parity` scores through the f32 working copy, so it is blind to the pool.

## Why
Free-running greedy chats still part from the f32 stream (after 237-654 tokens): a near-tie flips
and the continuation differs, which is why 396's greedy-chat check looked like a failure. Teacher
forcing is the comparison that separates a numeric near-tie from lost information.

## What to do
Gate: `tests/gdn_state_pool_real.rs` serves four long greedy streams (1024 tokens) on the f32 pool,
then scores that same stream teacher-forced through the pooled batched decode on both pools. Result:
top-1 99.63 %, mean KL 1.08e-4 nats, better than the 98 % / 1.5-3e-3 spread of the other precision
changes.

Verify: the verify and commit kernels address f32 rows in place. Rather than a bf16 build of each:

- a bf16 pool is gathered into f32 staging rows (the existing bf16 gather);
- the unchanged kernels run on the staging with identity row ids;
- the last-row state is scattered back (the existing bf16 scatter);
- a rejection gathers the f32 snapshot rows and scatters them rounded.

Cost: two small copies a layer. Greedy speculation agrees with plain decoding at the same precision
to the existing 12-token criterion (a plain step rounds the state after every token, a verify round
only after the last row).

## Evidence
Measured effect (gpu1, brain's default speculation): c=64 at 8K +6.6 %, others within noise, 32K x
16 worse (71 -> 56 tok/s) with the default speculation on; with speculation off the precision is
neutral there. The default speculation itself costs that cell 8x in TTFT (open in the
serving-runtime ledger).
