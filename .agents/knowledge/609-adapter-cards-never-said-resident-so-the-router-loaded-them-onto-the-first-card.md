<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 609. Adapter cards never said resident, so the router loaded them onto the first card

**Lesson:** a residency flag computed from the instances that are built must be asked under the instance's name, not
the listed name. An adapter is served by its base's instance, so "is an instance named `base:adapter` built?" is
always no. Observed 2026-11-27 on 16 GH200 servers: all 496 adapter cards said `resident: false`, also on the
upstream where the base was resident. A batch of 105 requests for one 8B adapter then sat as 64 on upstream 0 (a card
holding the 27B, which evicted it to build the 8B, then rebuilt the 27B five minutes later) and 41 on upstream 2 (a
training card, 0 tok/s) until all 105 timed out at the client's 480 s. 37 model evictions in 21 minutes on 12
serving cards.

## Why

- `apiserve::models::resident_models` collected `placements[].key.model`, the base names; `card_of` asked it for the
  adapter's own name.
- With no resident upstream, `route::Fleet::pick` marked no view active, and `affinity::choose_replica` fell to
  `free.first()`: the lowest-index open upstream, whatever it ran. That rule is right inside one process (fill the
  next replica of the same model in turn); between servers that hold different models it means "evict whatever the
  first card runs".
- Each adapter id counted its own load against the card's `decode_slots`, so 53 adapters of one base on a card were
  53 batches of 64 to the router, while the card decodes one.

## Fix

A card names its `slot_group` (the model whose instance serves it: `Executor::slot_group`, the instance key of a
generation) and is `resident` when that instance is built. The router counts load per slot group (an adapter and its
base share one bound per upstream) and, when no upstream with room has the model loaded or may take a spill, loads it
on an idle upstream, else on the one known to hold the fewest requests (the router's, or what its health reports),
never simply the first. Specs: `crates/apiserve/tests/it/api.rs::an_adapter_card_is_resident_when_its_base_is_and_names_the_base_as_its_slot_group`,
`crates/apiserve/tests/it/route.rs::a_model_resident_nowhere_goes_to_an_idle_upstream_not_the_first_open_one`,
`adapters_of_one_base_share_one_load_bound_on_an_upstream`,
`a_burst_for_a_model_hot_on_one_upstream_of_a_busy_fleet_goes_to_its_least_loaded_cards`.

**How to apply:** when a listed name and the thing that does the work differ (adapter and base, replica and model),
every per-name signal (residency, load, bounds) must be read through the mapping, and a fallback "first free" choice
needs a reason it is cheap in the setting it runs in.
