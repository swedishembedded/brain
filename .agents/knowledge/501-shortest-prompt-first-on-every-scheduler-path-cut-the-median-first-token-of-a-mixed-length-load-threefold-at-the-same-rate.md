<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 501. Shortest prompt first on every scheduler path cut the median first token of a mixed-length load threefold at the same rate

**Lesson:** the order prompts are read in is a scheduler policy, so it lives in one place for every decoder. Reading
the prompt with the fewest rows left first lowers the median first token without costing throughput, and the oldest
prompt needs a guaranteed share or a stream of short prompts starves it.

## What happened

`model::serve::Scheduler` had four orders: the sequential path (whole-prompt decoders such as Qwen3 and the 35B-A3B
MoE) read the oldest prompt first, the batched path shared rows oldest first in equal slices, the mixed path (the
hybrid 27B) read the shortest first with a floor for the oldest (knowledge 400), and the 27B's background reader chose
its own (short prompts first, a long one after eight of them). With equal prompt lengths all of them are first come,
first served; with mixed lengths a 2K prompt behind a 32K one waited for its whole prefill.

Every path now takes `Scheduler::allot_prefill`: the prompt with the fewest rows left to compute (a prompt less the
prefix it adopts from the cache) first, ties to the older; the oldest is given an eighth of the iteration's rows where
the decoder reads prompts in parts (the mixed rounds keep their engine's `min_prompt_rows`), and where it reads whole
prompts it earns one prefill round of rows for every iteration it is passed over and is read once they cover its
prompt. Every waiting request that fits is admitted at once, so the order sees all of them. The background reader
reads its jobs in the order the scheduler hands them.

`brain perf run serve`, 16 clients, prompts of 2048, 8192 and 32768 tokens mixed, 48 requests, 128 output tokens,
gpu3, two A/B pairs (before 2385f375f, after the change):

| model | req/s | TTFT p50 | TTFT p95 | ITL p95 |
|---|---|---|---|---|
| 35B-A3B MoE | 0.721-0.723 -> 0.722-0.724 | 2719-2724 -> 954-956 ms | 17.3-17.4 -> 15.2 s | 662 -> 660-661 ms |
| Qwen3-4B | 1.446-1.454 -> 1.454-1.457 | 1393-1396 -> 224-229 ms | 8.03-8.08 -> 6.81-6.85 s | 222 -> 220-221 ms |
| 27B (mixed path, order unchanged) | 0.765-0.772 -> 0.768-0.771 | 1195-1265 -> 1207-1208 ms | 14.3-14.5 -> 14.4 s | 353-355 -> 352-353 ms |

## Why

A burst of N prompts read one after another finishes prompt k at k times the mean prompt time; ordering them by
length makes the short ones' k small without moving the last one's finish, which is the total work either way.

## What to do

- A new prefill path takes `allot_prefill`; a decoder never orders the prompts itself.
- Evidence: `crates/model/tests/it/serve_prefill_order.rs` (the first tokens come in order of prompt length on the
  whole-prompt, sliced and batched paths; a long prompt behind a short one arriving every iteration is read within a
  bound).
