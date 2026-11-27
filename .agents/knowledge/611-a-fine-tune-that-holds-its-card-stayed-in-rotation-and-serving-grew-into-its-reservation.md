<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 611. A fine-tune that holds its card stayed in rotation, and serving grew into its reservation

**Lesson:** a card running a workload that does not lend it is not a serving card, and a reservation a run fills as
it goes is not free memory, however a live measurement reads it. Observed 2026-11-27 on 16 GH200 servers, five
of them running LoRA fine-tunes: each training card held 13-41 queued requests at 0 tok/s for the whole window
(the router saw them as idle, nothing of its own in flight and a stale or empty queue), and one fine-tune died
with `cuMemAlloc` out of memory (12 min of training lost) when the router sent 27B requests to its card and the
card's memory went from 89 to 94 GB.

## Why

- `/health` said nothing about a held card, and `route::Member::routable` only knew healthy, draining and
  stalled.
- A workload's reservation (its footprint) is charged to the card's budget when it is placed, but the run
  allocates it over its first minutes. `ResidencyManager::refresh_accelerator_capacity` set the card's total to
  `used + free_now`: the part of the reservation not yet allocated was counted in `used` and again in `free_now`,
  so the card "grew" by it and serving (a 27B build, its KV) was placed in it. `replan_starved` likewise offered a
  re-plan the whole usable card, the running workload's reservation included.

## Fix

The executor tracks the devices of each running workload group; `Stats::serving_held` is true while every device
the server computes on runs one that has not lent it (`capability::lease`). `/health` says `serving_held` with
`free_slots` 0, and the router takes such an upstream out of rotation until a probe says otherwise. While a workload
holds a card, a capacity measurement may shrink it but never grow it, and a re-plan is offered the card less the
reservation. Specs: `crates/residency/src/executor.rs::a_card_running_a_workload_that_does_not_lend_it_is_held_from_serving`,
`crates/residency/src/manager.rs::serving_never_grows_into_a_running_workloads_reservation`,
`crates/apiserve/tests/it/route.rs::an_upstream_whose_cards_a_workload_holds_takes_no_request_until_it_ends`,
`a_burst_never_lands_on_cards_a_fine_tune_holds`.

**How to apply:** a live "free memory" reading is only as good as the accounting it is added to: wherever a charge
is a promise rather than an allocation, the two overlap and must not be summed.
