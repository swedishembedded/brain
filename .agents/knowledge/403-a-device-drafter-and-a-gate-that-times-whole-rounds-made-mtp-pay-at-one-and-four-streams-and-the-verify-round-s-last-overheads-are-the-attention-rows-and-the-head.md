<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 403. A device drafter and a gate that times whole rounds made MTP pay at one and four streams; the verify round's last overheads are the attention rows and the head

Question (knowledge 395): the FP8 release's MTP head is exact but lost to plain decoding. What does it take for speculation to pay on the goal matrix (random prompts, greedy,
256 new tokens), and what is left?

## What was wrong, in the order it hid

1. **The gate never ran a round.** The cost model priced a round as the plain step plus a per-row and a per-proposal price, learned only from rounds that ran. One warm-up of
   eight sequences left 5.5 ms a row; every later round was declined, so nothing re-measured it (8K x 1: zero rounds in 256 tokens). It also assumed a line through the rows,
   and the first extra row costs a kernel switch (2-row round 14.4 ms against 11.5 for one row, the 4-row 16.9). `CostGate` now times whole rounds per batch bucket and
   window and runs the window with the most tokens a second; every 16 steps it re-measures a neighbour (or plain). On these prompts the first drafted token is accepted
   ~0.9 of the time and the second almost never (tokens a round: k=1 1.96, k=2 2.0, k=3 1.7), so k=1 or 2 wins: 8K x 1, arms (tok/s): plain 77, k=1 124, k=2 107, k=3 79.
2. **The prompt feed cost 2.8 s per 8K prompt** (host embedding rows, host norms, a projection read back and uploaded, 128-row rounds, 40 ms each); the prefill gate had been
   skipping it, so a prompt's sequence drafted from an empty head and nothing was accepted. The feed is now a device pipeline (rows uploaded, `rms`, `eh_proj`, the layer's
   prefill chunk at 1024 rows): 8K prompt ~0.1 s; it runs after the first token (deferred), so it is not in the TTFT.
3. **A draft proposal cost 3.3-5 ms** (four host round trips per chained proposal). One round is now one recorded graph per (sequences, catch-up rows, proposals, attention
   buckets): gather the pending hidden rows the target step itself wrote (`row_scatter` at the end of every recorded decode and verify step), norms, `eh_proj`, the layer, the
   last row, the shared head, argmax, chained for k proposals, one readback. k=1: 1.3 ms of GPU (the bf16 head 0.69 of it, six small rms kernels 0.17).
4. **The GDN verify kernel spilled.** `brain_gdn_verify_pool` (one 384-thread block per key head) needs 128 state registers beside the conv windows and gates in 168:
   391 000 local-memory requests in a 32-block launch (ncu), 47 us a layer at two rows against 12 for the plain split decode. The verify now takes the split decode's form
   (conv kernel + a recurrence block per value head on a shared-memory tile): 47 -> 19 us a layer, the verify round 14.4-15.0 -> 12.8 ms. Bit-identical (the test compares with repeated
   single steps).

## Where the round is now (nsys, 8K x 1, gpu0)

draft 1.3 ms + verify 12.8 + commit ~0.1 (a replay of 48 small copies, only when something was rejected) for ~1.9 tokens: ~7.6 ms a token against 11.7 plain. The 2-row verify
costs 1.3 ms more than a plain step: attention reads the sequence's KV once per row (32K: 2.6 ms against 1.5, so 1.1 ms more; 64K twice that), the bf16 GDN projections and
the head take the library split-K path at two rows (+0.4 ms), the per-row gates. What would take them: a multi-row decode attention (the rows of a sequence share one KV read: the
tensor-core tile has 16 query slots and a group of 6), a decode-width bf16 GEMV for 2-4 rows, an fp8 copy of the head for the draft only.

## Served (same client, 256 new tokens, 1 / 4 streams, `--speculative off` -> `mtp`)

See the ledger (`serving-runtime.md`, item 18); at steady state (1024 new tokens) 8K x 1 is 81.6 -> 94-99 tok/s (TPOT 11.7 -> 9.6-10.0) and 8K x 4 (768 new) 246 -> 276.
