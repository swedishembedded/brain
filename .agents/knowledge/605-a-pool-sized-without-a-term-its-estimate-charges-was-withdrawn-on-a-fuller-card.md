<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 605. A pool sized without a term its estimate charges was withdrawn on a fuller card

**Lesson:** the arithmetic that sizes a plan to a share and the arithmetic that charges the plan against that share
must be the same function. Two sums of "the same" demands drift: one gains a term (an adapter bank, a worst-case
scratch) and the other does not, and the plan sized to fit is refused as too big. It hides on a roomy card and
appears as soon as the card has less free.

## What happened

On 2026-11-26 servers on GH200 cards where a leaked context of a killed server held 5.4 to 6.4 GB logged
`UNSERVABLE Qwen/Qwen3-4B ... needs 91869 MiB ... more than any device on this host has` and withdrew the model; the
same command on a clean card served it, and `--max-batch 32` instead of 64 made it servable.

## Why

The budgets were already built from free bytes (`serving::Machine::probe`). But `QwenResident`'s context and fleet
pool were sized with `PoolProfile::fits_budget`, which charged no adapter bank and only the fused GPU attention
scratch, while `estimate` charged the bank (`--serve-adapters --adapter-targets all`, sized over 64 rows) and the
larger of the fused and never-fused scratch. At 64 rows the 1M-token pool fitted the sizing on an 86 GiB share and
the estimate came out at 87.1 GiB. The pool also shrank by halving, so a share just short of 1M tokens fell to 512K
and left 30 GiB of an 80 GiB share unused.

## What to do

`serving_demands` in `crates/catalog/src/resident_llm.rs` is the one footprint: the estimate, `unfit_reason`, the
batch plan, the context tiers and the pool search call it. The pool is the largest that fits
(`model::memplan::largest_fitting`), down to one context; only a share that cannot hold that is refused, naming
the arithmetic. Spec: `a_wide_batch_with_an_adapter_bank_sizes_its_pool_to_the_card_share`.
