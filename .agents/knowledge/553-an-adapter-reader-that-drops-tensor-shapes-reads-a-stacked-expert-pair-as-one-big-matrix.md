<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 553. An adapter reader that drops tensor shapes reads a stacked expert pair as one big matrix

**Lesson:** a reader of factor tensors must look at the shape the file records, not only at the element count.
`checkpoint::st::StModel` kept every tensor as flat `f32` data and dropped the header's shape, so the device-adapter
reader could only check that a factor's length was a multiple of the rank. A stacked expert pair (`A [E, r, in]`,
`B [E, out, r]`, one low-rank pair per routed expert) passes that check, and `AdapterSite::dims` then reads it as one
`[E * out, E * in]` linear: a shape that exists nowhere in the model. Some consumers refused it later, by name or by
size, with a message about the wrong thing; none could tell that it was a different kind of pair. `StModel` now carries
`shapes`, and `lora::device_adapter` classifies every pair by its header shapes: 1-D or 2-D factors are a linear pair,
3-D factors over one expert count at the card's rank are a stacked pair, and anything else is refused naming both
shapes. The linear readers (`read_adapter`, `parse_adapter`, `adapter::file::AdapterFile::read`) refuse a stacked
pair by name. Only `read_adapter_with_experts` / `parse_adapter_with_experts` accept it, into
`DeviceAdapter::experts`, so a consumer that knows only `DeviceAdapter::sites` never silently drops one.

## Evidence (2026-11-22)

- `crates/model/tests/it/adapter_expert_sites.rs`, `malformed_or_misdirected_stacked_adapters_are_refused_by_name`:
  a well-formed stacked file is refused by `read_adapter` and `AdapterFile::read`. A 3-D/2-D mix, two expert counts,
  a rank that differs from the card's, and 4-D factors are each refused naming the bank.
- Same file, `a_trained_stacked_adapter_round_trips_and_serves_beside_the_bank`: the shared writer
  (`save_lora_with_lineage` with `Model::adapter_expert_sites`) stores `A [E, r, in]` and `B [E, out, r]`, and the
  experts-aware reader gives back the factors bit for bit.
