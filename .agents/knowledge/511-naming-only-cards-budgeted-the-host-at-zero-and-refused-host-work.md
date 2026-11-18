<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 511. Naming only cards budgeted the host at zero bytes, and host work was refused as too large for every device

**Lesson:** whether the CPU computes and how much host memory there is are two facts. A placement budget that
stands for both (a zero budget meaning "no CPU compute") refuses every byte of host work and leaves host charges
uncounted. Keep the host's real RAM as its budget and exclude compute with a flag the placement asks.

## What was reported

`brain serve --training --device gpu3` refused `brain/train`'s `check_dataset` with `brain/train[host] (256 MiB)
exceeds every device's budget even with nothing resident`; `--device gpu3,cpu` worked. The check reads a tokenizer
and a dataset in the serving process: it has no other place to run.

## Cause

`serving::Machine::probe` gave `Device::Cpu` a budget of `0` when `--device` left the CPU out, so that nothing would
be placed there. `trainjob` reserves host work as `MemCost::new(0, bytes)`, the same shape as a model that computes
on the CPU, so `place::could_ever_fit` found no device and the claim failed as `TooLarge`. The same zero budget had
already needed two workarounds: a multi-device model's staging bytes and a hosted worker's own memory were left
uncharged when the host budget was zero (`charges_multi_host`, `process_host_charge`), and a Warm demotion could
never fit there.

## Fix

- `Device::Cpu` is budgeted at the host's RAM (`Machine::pool_ram`) whatever `--device` says;
  `Budgets::exclude_cpu_compute` marks the CPU as computing no model (`serving::budgets_for`).
- `MemCost::host_work` declares host-side work. `Budgets::hosts(cost)` is the one question every placement path
  asks before choosing `Device::Cpu` (`pick_device`, `spill_to_host`, `could_ever_fit`, eviction, `fit_devices`,
  `supported_devices`, the one-shot plan's host tier): yes when the CPU computes, else only for host work and for
  instances that hold nothing.
- `brain/train`'s host reservations (`check_dataset`, `merge`, `init_model`, a `full_train` that only validates)
  are host work.
- With the host budget real, staging bytes, a worker's own memory and Warm copies are charged against host RAM on a
  server given only cards, as they always were on one given the CPU.

Tests: `trainjob` `a_dry_run_runs_on_the_host_whatever_devices_the_server_was_given` (one card, two cards, the CPU
alone; it fails with the reported message on the old budget), `residency` `host_work_is_placed_on_the_host_whatever_
devices_are_named`, `serving` `host_work_fits_the_budgets_of_every_device_list`.
