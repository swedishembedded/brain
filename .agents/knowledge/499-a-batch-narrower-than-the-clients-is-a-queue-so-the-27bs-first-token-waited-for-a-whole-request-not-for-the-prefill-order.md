<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 499. A batch narrower than the clients is a queue, so the 27B's first token waited for a whole request, not for the prefill order

**Lesson:** before blaming the order in which prompts are read in, check how many sequences the engine admits. A
request beyond `max_batch` waits in the scheduler's queue until a running request has read its prompt AND produced
every output token, and that wait is all of its time to first token. A default batch fixed for one old placement
(eight slots, sized for two 24 GiB cards) became a queue on every card that holds far more.

## What happened

Question: `brain perf run serve`, 8K prompts, 128 output tokens, 16 closed-loop clients, one GH200: the FP8 27B's time
to first token was p50 7.4 s / p95 12.0 s against the 35B-A3B MoE's 0.67 s / 10.2 s, although both read prompts in at
about the same rate. Is the 27B's mixed-round prefill order the cause?

No. The 27B ran with `max_batch` 8 (`catalog::resident_qwen35::DEFAULT_MAX_BATCH`), the MoE with 64 (its planner takes
the widest batch that leaves one sequence a 32K context). Eight of the sixteen clients waited in the queue for a whole
request each. An nsys timeline of the 27B run shows it: about eighteen mixed rounds (eight 8K prompts) and then about
a hundred decode-only steps, four times over, where the MoE's waves of sixteen pipelined into one prompt per iteration.
The minimum first-token time of the 27B run was 3.0 s, never one prompt's prefill.

Measured on gpu3, the same binary, `--max-batch` given (perf serve, seed 7):

| load | batch | req/s | TTFT p50 | TTFT p95 | ITL p95 |
|---|---:|---:|---:|---:|---:|
| c16 x 8K | 8 | 1.16 | 6976 ms | 11182 ms | 278 ms |
| c16 x 8K | 16 | 1.33 | 1146 ms | 9480 ms | 288 ms |
| c16 x 8K | 64 | 1.33 | 1146 ms | 9459 ms | 289 ms |
| c64 x 8K | 8 (shared run) | 1.18 | 27304 ms | 48594 ms | 293 ms |
| c64 x 8K | 64 | 1.47 | 19667 ms | 37930 ms | 317 ms |
| c16 x 32K | 8 (shared run) | 0.33 | 24597 ms | 43885 ms | 356 ms |
| c16 x 32K | 64 | 0.33 | 22407 ms | 43614 ms | 363 ms |

## Why

- The scheduler admits `min(max_batch, memory pressure)` sequences; everything else is `waiting`, and nothing of a
  waiting request is computed. With a closed-loop load the waiting half is served only as the running half finishes,
  so its first token comes a whole request (prefill of eight prompts and 128 decode steps) after it arrived.
- The order of the prompts inside a round decided almost nothing here: with equal prompt lengths, the mixed round's
  shortest-remaining-first allotment is first come, first served.
- A wider batch decodes each step slower (ITL p50 13 -> 48 ms at 64 rows), but more requests decode at once, so the
  served rate rose and no request waited outside.

## What to do

- The 27B's default batch is now chosen like the MoE's, by one shared rule
  (`catalog::serve_defaults::MAX_BATCH_DEFAULTS`): the widest of 64/32/16/8/4/2/1 that leaves one sequence the 32K
  context floor, after the KV tier and before speculation and the prompt cache. The sequences share one KV pool, so a
  slot costs its decode activations and recurrent-state row, not a context.
- When a serving comparison shows one engine's first token far behind another's, compare `peak_running_sequences`
  and `max_batch` in the perf artifact before anything else.

## Evidence

`catalog::resident_qwen35::tests::config_defaults_size_the_context_and_batch_the_requests` (two roomy cards take the
widest batch, reported with its reason) and `the_default_batch_gives_way_to_keep_the_context_floor` (a card on which
the widest batch would squeeze the context under the floor takes a narrower one; an operator's batch is kept).
