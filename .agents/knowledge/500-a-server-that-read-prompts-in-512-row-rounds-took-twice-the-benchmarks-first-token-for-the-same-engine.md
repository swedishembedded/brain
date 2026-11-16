<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 500. A server that read prompts in 512-row rounds took twice the benchmark's first token for the same engine

**Lesson:** a benchmark measures what the server does only when both size the engine with the same rule. `brain
serve` built the Qwen3 engine with a 512-row prefill round (a constant kept from a wgpu buffer limit of the unfused
attention), while `brain perf run serve` built it with a whole prompt per round. Through `brain serve` 16 x 32K prompts
had a first-token median of 20.8 s, where perf measured 10.5 s for the same engine.

## What happened

The 512-row cap protected the unfused prefill attention, whose score slab grows with the square of the round. Every
GPU backend fuses that attention, and there the scratch does not grow with the round; the cap stayed anyway, so a
32K prompt took 64 rounds, each with its per-layer dispatch overhead.

Both now take `qwen3::serve::prefill_round_rows`: the measured round of `model::serve::prefill_round_rows_for` (4096 rows
on tensor cores, 256 without) where the prefill attention is fused, the 512-row bound elsewhere. `brain serve` holds a
planned round under 512 rows when the engine lands on a device that fuses nothing, and charges the larger of the two
forms before placement.

Qwen3-4B, one GH200 (gpu3), 128 output tokens, 16 clients; base 5e45c6ff6 against the rule, two A/B pairs:

| path | prompts | before | after |
|---|---|---|---|
| `brain serve` (HTTP, streamed) | 16 x 32K | 0.50 req/s, TTFT p50 18.0-18.3 s, p95 30.8-31.3 s | 0.64 req/s, 14.4-14.5 s, 24.5-24.8 s |
| perf serve | 16 x 8K | 3.32 req/s, TTFT p50 417-420 ms, ITL p95 34 ms | 3.25-3.27 req/s, 216-218 ms, 215 ms |
| perf serve | 16 x 32K | 0.64 req/s, TTFT p50 10.3 s | 0.59 req/s, 11.1-11.3 s |

## Why

- Perf's whole-prompt rounds are about 8 % faster at 32K on the dense 4B, but a whole-prompt round's activations grow
  with the context: on the 8B at a 32K context a 94 GB card then holds half the KV pool for the fleet of histories
  (`resident_llm::tests::the_real_8b_on_a_big_card_gets_the_fleet_pool`). The bounded round is the one rule both sides
  can take.
- With rounds of one prompt's worth, decode runs between them: the 8K median first token halves and a running
  sequence's token interval at p95 becomes one prompt's round.

## What to do

- Size the server's engine and the benchmark's engine by the same function; a constant in one of them is a
  measurement of something else.
- Open: a round width measured for the dense Qwen3 engine (8K and 16K rows were not measured).
