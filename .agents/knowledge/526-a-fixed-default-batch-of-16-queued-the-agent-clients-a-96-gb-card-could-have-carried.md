<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 526. A fixed default batch of 16 queued the agent clients a 96 GB card could have carried

**Lesson:** A serving default that depends on the card is planned from the card, by the one planner the other
families use (`serve_defaults::MAX_BATCH_DEFAULTS` through `given_or_chosen`), and logged with its reason. A constant
chosen for a small card is a queue in front of every larger one, and a queue shows up in a client as a timeout, not as
a slow server.

## What happened

`brain serve --device gpu0` of Qwen3-4B bf16 with a LoRA adapter on a GH200 96 GB took a fixed `--max-batch 16`
(`QwenServeConfig::max_batch`), though Qwen3.8 already planned its batch. Aggregate decode of 512-token completions:

| Concurrent sequences | `--max-batch` 16 (default) | `--max-batch 64` (explicit) |
|---|---|---|
| 16 | 1526 tok/s | |
| 32 | 1531 tok/s (16 queue) | 2581 tok/s |
| 64 | | 4451 tok/s |

Agent clients run 24-48 sequences, so requests beyond 16 waited for a whole running request, and a bench gateway hit
HTTP 504 on about 15% of long trials.

## What fixed it

`QwenServeConfig::max_batch` is an `Option`. Unset with a card budget known, `QwenResident::resolve_max_batch` takes
the widest of 64, 32, 16, ... for which `memplan` places weights, the adapter bank, the activations of that many
rows and a KV pool for one sequence at the context floor (32K, or the checkpoint's window when shorter, or the
operator's `--ctx`). The context and the fleet pool are then sized for that batch, and the startup log says
`brain: <id>: default --max-batch N: why`. Without a budget (a library caller) it stays 16. An explicit
`--max-batch` is taken as given.

The regression test is `an_unset_max_batch_is_planned_from_the_card` in `crates/catalog/src/resident_llm.rs`.

Open: a Qwen3 resident is not re-planned when a co-tenant later shrinks the card; only the Qwen3.8 residents are
(`ResidentModel::replan`).
