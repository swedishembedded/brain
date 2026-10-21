<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 351. A prompt resumed from a cache entry inside a recurrent chunk drifted fifteen times more than the slice grid does, and single-stream greedy decoding is reproducible

Question: do two runs of the same binary on the same prompt at temperature 0 and concurrency 1 give the same reply (roadmap 4.4), and
does resuming from the prefix cache give what a cold prefill gives? Every figure is VERIFIED-BY-RUN on the served 27B Q8_0 GGUF, int8 KV,
one GH200, card 1.

## Single stream is reproducible; the first "non-determinism" was a comparison error
- Replaying recorded trajectories twice on the unpatched 0cc4bf64c at concurrency 1 (47 requests, thinking on, temperature 0), the
  reassembled replies (reasoning, content, tool-call arguments) were identical 47 of 47. Comparing the stream chunk by chunk reported 0 of 47,
  because the server's chunk boundaries move with timing: compare reassembled text, never chunks.
- Same prompt 4-6 times on one server (short prompt, 2.4k and 3k prompts; thinking on and off; prefix cache off, and on across
  three cold server starts): the same token ids at every step.
- At concurrency 8 the same request sequence gave 17 of 108 identical replies: the batch composition changes with arrival timing and the
  engine's arithmetic is not batch-invariant (the fused split-key attention picks its split count from the batch; GEMV tiers round
  differently; knowledge 318). Not claimed, not changed.
- `logprobs` at temperature 0 reports 0.0 for every token (the log-probability under the greedy distribution), so it says nothing about
  numerical agreement. Use temperature 1 with a fixed `seed`: cold against cold is then identical bit for bit in tokens and logprobs.
- Regression test: `single_stream_greedy_decoding_is_reproducible` (two fresh engines, same tokens; one prompt prefilled twice, same hidden bits).

## Resuming from the cache drifted, and where it came from
Teacher-forced comparison at the same positions (temperature 1, fixed seed, sampled-token log-probability, up to the first step where the
sampled tokens differ; 8 prompts of 2.3k tokens plus a 1.1k-token next turn, thinking off / on):

| prefill | mean abs dlogprob off / on | sequences identical (of 8) off / on |
|---|---:|---:|
| cold, slices of 256 against slices of 512 (no cache) | 0.005 / - | 5 / - |
| resumed from the previous turn's entry, checkpoint inside a 64-token chunk (the 16-token-tail commit) | 0.130 / 0.037 | 0 / 0 |
| the same with the unpatched binary's whole-prompt entry (thinking off) | 0.094 / - | 0 / - |
| resumed, checkpoint on a 64-token boundary (this fix) | 0.0088 / 0.016 | 3 / 1 |
Max single-step difference stays 0.4-0.6 nats in every row with a divergence (a near-tie flip). The drift was not new to the 16-token tail:
the old whole-prompt entry had it too. Cause: a resume starts the Gated-DeltaNet chunk recurrence in the middle of a 64-token chunk, so
the chunk partition differs from the cold prefill's; the state and KV copies are exact (the snapshot and `copy_rows` move words).

Fix: the checkpoint is rounded down to a multiple of 64 (`CHECKPOINT_ALIGN`), so a resumed prompt walks the same chunks as a cold one
(at most 63 more tokens recomputed). Test: `a_resumed_prefill_is_bitwise_a_cold_one` (tiny GGUF engine, hidden row equal bit for bit).
Aligning the first resumed slice to the 512-token grid as well changed nothing measurable and was not kept.
Residual drift (about the size of re-slicing a cold prefill) remains on the 27B; bit identity of a resumed against a cold reply is not claimed.
