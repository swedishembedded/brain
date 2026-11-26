<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 603. Two models on one fleet swapped every card back and forth: the router saw neither residency nor per-model slots

**Lesson:** a router in front of servers that load models on demand must know, per model, where the model is loaded
and how many sequences of it a server decodes. A server's total capacity is not a model's: `/health`'s
`decode_slots` sums every model the server could load, and a card that would have to load the model evicts the one it
holds.

## What happened

On 2026-11-26, 16 `brain serve --openai` servers (one per GH200, `--max-batch 64`, Qwen3.8-27B-FP8 and Qwen3-4B
both servable) sat behind two routers. Two agent workloads ran at once: up to 512 concurrent conversations with the 4B
through one router and up to 384 with the 27B through the other, both routers over all 16 servers.

- Over about two hours each server built the 27B 1 to 8 times (49 to 80 s a build) and the 4B about as often, alternating:
  a card holding one model was sent a request for the other, evicted what it held and built the other.
- Every server's `/health` reported `decode_slots` 1482 (the sum of the batches of every registered model) against 64
  sequences per model, so the router's balance term (a cached prefix is worth `slots / 8` requests) let a prefix pull
  up to 185 requests of one model onto one card, and on the 4B workload's start all of it went to 4 of the 16 cards.
- In the 4B workload, 181 of 60026 model calls waited more than 100 s for their first token (up to 417 s, against
  0.32 s at the median), and 334 of 8208 conversations held a request stuck for more than 100 s: 44.8 of about 165
  slot-hours went to those conversations, most of which ran into their own time limit.

## Why

`route::Fleet::pick` built each upstream's `ReplicaView` with `slots` from `/health` and `active: true`, so the
replica policy (`residency::affinity::choose_replica`), which prefers a replica where the model is built, saw every
upstream as built and as wide as all its models together.

## Fix

A model card on the OpenAI and OpenRouter dialects says `decode_slots` (that model's batch on each replica) and
`resident` (built on a device now). The router bounds each model's requests per upstream by the card's slots (and the
upstream's total by `/health`'s), and passes `active = resident` to the policy, so a model's requests go to upstreams
where it is loaded while one has a free slot for it. Tests: `crates/apiserve/tests/it/route.rs`
(`an_upstream_takes_at_most_its_models_own_slots_of_that_model`, `a_request_goes_where_its_model_is_resident`),
`crates/apiserve/tests/it/api.rs` (`model_card_says_how_many_sequences_the_model_decodes_and_whether_it_is_resident`).
