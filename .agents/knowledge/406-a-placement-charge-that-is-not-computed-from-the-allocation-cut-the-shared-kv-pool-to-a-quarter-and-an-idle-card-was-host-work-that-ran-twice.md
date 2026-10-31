<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 406. A placement charge that is not computed from the allocation cut the shared KV pool to a quarter, and an idle card was host work that ran twice

Question: brain's default flags (MTP speculation while at most two sequences run) served 8K x 64 at a 30-32 s median first token and 32K x 16 at 37-45 s,
against 1.7 s and 4.2 s with speculation off. Why, when speculation never runs at those batch sizes?

Measured, not reasoned: the step timeline (NVTX `serve.step` / `serve.mixed`) showed seven prompts read, then 199 decode-only steps while nine prompts of
the same wave waited, then the next few: admission stalled on KV blocks. The server's own startup line named the cause: `default --ctx 204800 ... one KV pool
of 204800 tokens` with the drafter loaded, 724,992 without. The planner charged the MTP head's attention layer `window_slots x cap` rows (the one-window
layout's formula: 64 slots x the whole context), while `ShardCaches` gives it, in the paged layout, exactly what every full-attention layer has: one layer of
the shared pool. The charge is now `layer_state_bytes(cfg, Full, shape)`, the function that charges the other layers, and the pool stays at 737,280 tokens.

Rule: a component's memory charge is computed by the same function (or from the same layout value) as its allocation, never restated beside it; a test
asserts that the charge does not grow with something the allocation does not depend on (here: batch and context).

Second finding of the same profile, host side: the card was idle 2.45 s of a 29 s 8K x 16 cell (the reference: 1.22 s), in 30-60 ms gaps before every
prompt. CPU sampling inside the gaps (nsys `-s cpu`, samples of the serving thread whose timestamps fall in a gap) showed `memcmp`, `memcpy` and allocator
frames: the tokenizer's merge loop cloned two strings for every pair lookup, and every prompt was tokenized twice, once to screen it and once to admit it.
Both fixed (id-keyed merges; take free slots before screening). Method worth reusing: charge each instant to the earliest-started active kernel when
comparing kernel time across engines, or programmatic-dependent successors count their wait as their own time (brain's combine kernel showed 114 us a
launch that was all waiting).
