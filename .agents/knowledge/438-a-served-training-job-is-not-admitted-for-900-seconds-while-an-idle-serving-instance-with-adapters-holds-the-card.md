<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 438. A served training job is not admitted for 900 seconds while an idle serving instance with adapters holds the card

**Lesson:** `brain serve --training` is documented to take a card by evicting idle serving instances. Twice
on a single Qwen3.8-27B-FP8 server (gpu3, `--watch-adapters`, `--adapter-targets all`) a `brain/train` job
was refused with "not admitted within the deadline (queue depth 1, deadline 900000 ms)" although nothing was
running on the card. A server that had not yet answered any request admitted the same job at once.

## What happened

- Run 1: a 2-step job trained and published an adapter (3m36s). The adapter was dropped into the watched
  directory and a bench run sent 7 requests through it. The next 40-step job waited 900.2 s and was refused;
  `nvidia-smi` showed 86 GB resident and 0% utilisation, and the only connection to the port was the client.
  Restarting the server and resubmitting trained all 40 steps (3859 s).
- Run 2 (reproduction, same server, A2 adapter served and 160 requests answered earlier, no request in
  flight): a 2-step job of the same dataset waited 900.09 s and was refused; 86 GB resident, 0% utilisation.

## Root cause (validated) and fix

The served model (a Qwen3.8 base with an adapter bank) is a **multi-device** resident (`activate_multi`); the
train job is a single-device workload. `ResidencyManager::claim` already frees a card for a single-device job by
dropping an unpinned multi-device resident (`evict_multi_to_fit`), but the dispatcher only calls `claim` for a
group that `ResidencyManager::placeable` accepts, and `placeable` planned eviction over single-device residents
only. With the served model resident and idle, `placeable` said "no" on every round, the job stayed queued
(depth 1) and was never claimed, so it waited out its admission deadline. A server that had answered no request
had not built the model yet, so the card was free and the job fit at once. The adapter bank, the watched
directory and the KV pool are not involved: nothing was pinned; the idle resident was simply invisible to the
placement filter.

Validated with a regression test (`a_single_device_job_is_placeable_when_an_idle_multi_device_resident_can_give_way`
in `residency::manager`): red before the fix at the "idle: ... can give way" assertion (the pinned assertion
passed), green after. The fix extracts the read-only victim planning of `evict_multi_to_fit` into
`plan_multi_victims`, and `placeable` consults it, so the filter and the claim cannot disagree. Generic: no model
named. A pinned resident (a lane running on it) still blocks the job, and a busy card is not offered.

Workaround before the fix: restart the server before a training job, or train on a card that serves no adapters.

Verified live 2026-11-13 (lane rollout): on a Qwen3.8-27B-FP8 server on one GH200 with `--training --serve-adapters` (three rank-32
adapters on all linears hot-loaded, about 600 requests answered across them, then idle) a `brain/train` job (2 steps, rank 16,
bf16 base) was admitted at once (stage frames within 20 s, the card handed over by evicting the idle instance), trained and
succeeded in 137 s; the next request rebuilt the serving instance (31 s). The stall does not reproduce.

