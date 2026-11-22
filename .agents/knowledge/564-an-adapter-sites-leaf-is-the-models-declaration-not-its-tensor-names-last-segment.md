<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 564. An adapter site's leaf is the model's declaration, not its tensor name's last segment

**Lesson:** a card's `targets` are site leaves, and the leaf is what the model declares for a site
(`LinearSite::leaf`, `ExpertSite::leaf`), so a reader that checks the card against the tensors must take the leaf
from that declaration. `device_adapter::adapter_from_model` derived it from the base name's last dot segment, which
holds only while every model names its sites `<..>.<leaf>[.weight]`. A bank of queries stored as rows of indexed
tensors breaks it: Inkling 2's `attn_res.attn_queries.0` is an `attn_queries` site whose last segment is `0`, so an
adapter the trainer had just saved over its depth queries did not read back ("card declares [.., attn_queries, ..]
but its tensors correct [0, 1, 2, ..]"). The writer (`save_lora_with_lineage`) now records `leaves` in the adapter's
config (pair base -> declared leaf, every pair); the reader checks the card against it, refuses a map that misses a
pair, names one the file holds no pair for or is not a map of strings, and falls back to the last segment only for a
file written before the map existed. The parsed adapter carries the map (`DeviceAdapter::leaves`, read through
`DeviceAdapter::leaf`), and the serving registry's admission check (`AdapterRegistry::fits`, which took the
second-to-last segment) asks it too, for the stacked expert pairs as well as the linear ones.

## Evidence (2026-11-22)

- `inkling2` `serve_adapters::an_adapter_over_every_site_kind_reads_back_with_the_trainers_factors` failed with the
  message above before the change and passes after it (every linear, merged and stacked expert pair bit-identical to
  the trainer's factors).
- `model` `adapter_contract::recorded_leaves_decide_what_a_pair_corrects` pins the map's refusals; the existing card
  contract tests (no map) pass unchanged.
