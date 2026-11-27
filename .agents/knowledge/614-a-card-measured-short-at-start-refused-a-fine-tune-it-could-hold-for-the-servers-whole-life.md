<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 614. A card measured short at start refused a fine-tune it could hold for the server's whole life

**Lesson:** what a card has free when a server starts is a fact about that moment, not about the card. Observed
2026-11-27 on 16 GH200 servers (brain a2e6c80a9, one per 96 GB card, `--training`): a LoRA fine-tune of about
46 GiB was refused with HTTP 400 (`train: this run needs about 46 GiB (asked for: Weights 15.42 GiB ...) and the
largest card has ...`) on a card whose residents (4B, 8B and 27B serving models plus adapters) were idle. The
operator had to restart the server to train on it.

## Why

- `serving::build_executor` gave the training door (`trainjob::TrainConfig::largest_card_bytes`) the bytes free on
  the card when the server started, less `--reserve-gb`. A server is started right after its predecessor is killed
  (the launcher waits for the process and its port, not for the driver to hand the memory back), so that figure can
  be far below the card's size, and it was fixed for the process's life. The door then refused, permanently, every
  run between it and the card's size.
- The placement had the same blind spot one layer down: a budget's "could ever fit" ceiling (`Budget::peak_total`)
  started at that same free figure and only grew when a later measurement saw more, so the claim of such a run read
  as `TooLarge` (fail now) instead of `Starved` (wait).
- A `Starved` claim, even when correctly classified, failed its queued jobs at once.

Placement itself was right: idle serving residents give way to a workload that needs their memory, and a request
for one that gave way waits and is served by a rebuild (`crates/residency/tests/it/recovery.rs`, green before and
after).

## Fix

`Machine` carries each card's size (`gpu_sizes`, from the same `nvidia-smi` probe, clamped to a `--vram-budget`
grant and `--limit-vram-total`). The training door is bounded by the largest card's size less the reserve
(`serving::executor::largest_card_bytes`), and every card's budget declares its size (`Budgets::declare_size`), so
"could ever fit" is the card's size while "fits now" stays the live measurement. A `Starved` claim keeps its jobs
queued (the model is listed in `Stats::unservable` with the reason, so a router sends elsewhere) and is claimed again
every round and capacity measurement; a job still short at its queue deadline is refused then, naming the shortage.
Specs: `crates/serving/src/executor.rs::a_card_short_at_start_refuses_only_what_its_size_could_never_hold`,
`crates/residency/src/budget.rs::a_card_short_at_start_could_still_hold_what_its_size_holds`,
`crates/residency/src/executor.rs::a_request_for_a_model_the_card_is_short_of_waits_and_is_served_when_the_memory_returns`,
`a_request_still_short_of_memory_at_its_deadline_is_refused_with_the_shortage`.

**How to apply:** a refusal is only as permanent as the figure it compares against. Compare "can this ever run" with
the device's size and "can this run now" with a live measurement, and let the second one delay work, never refuse it.
