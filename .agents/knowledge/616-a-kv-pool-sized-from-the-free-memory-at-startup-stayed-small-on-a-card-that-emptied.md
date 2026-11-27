<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 616. A KV pool sized from the free memory at startup stayed small on a card that emptied

**Lesson:** a default sized from a live measurement is only as current as the measurement. `Machine::probe` budgets a
card at the bytes free when the server starts (right for placement, #605), and `QwenResident` resolved its KV pool
from that budget once, at construction, for its whole lifetime. A server started while another process held most of a
card kept that small pool after the card emptied.

## What happened

2026-11-27, n547 gpu1 (no training): the Qwen3-4B pool was 12261 blocks of 16 tokens (196k tokens; the plan at about
a 27 GiB share) while 48 of 97 GB of the card were unused. Agent contexts of 5-30k tokens fit about 16 at once, so the
4B decoded at a width of 8-16 with dozens queued. The same plan on a whole 93 GiB share is a 1M-token pool (84 GiB
estimate). In the CPU harness of #613 (96 agent trials, 8 turns, max batch 64) the 196k pool runs a mean width of 16
over about 7.5k iterations and a 1M pool a mean of 55 over about 2.2k.

## Fix

`ResidentModel::fit_to_room`: before a cold build is placed, the manager offers the model the bytes free on each
card it may land on beside everything that stays resident (a running workload's reservation is charged there, #611),
and `QwenResident` grows a defaulted pool to the largest its own estimate (`serving_demands`) fits in that room. It
never shrinks there (a fuller card evicts as before); `replan`, asked when a claim is starved, re-sizes it to what the
card gives now. A pool the operator named, and a worker-hosted resident (its worker was sent the plan), keep theirs.
The context, and so the manifest, does not change. Specs:
`crates/residency/src/manager.rs::a_cold_build_grows_a_defaulted_pool_into_the_room_beside_what_stays_resident`,
`crates/catalog/src/resident_llm.rs::a_defaulted_pool_grows_into_the_room_a_build_finds_and_a_named_one_does_not`.

**How to apply:** a running instance's pool does not grow until it is built again (an eviction, a restart). Growing a
live pool needs a pool in segments; until then, size defaults at build time, not at startup.
