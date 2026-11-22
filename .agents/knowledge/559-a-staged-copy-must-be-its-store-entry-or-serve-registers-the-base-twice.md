<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 559. A staged copy must be its store entry, or serve registers the base twice

**Lesson:** `--stage-models` rewrites argument paths, and `brain serve` recognises the `--checkpoint` resident as the
model scan's entry by the weights path (`executor::same_checkpoint`). A rewritten path names the copy while the scan
finds the store entry, so the base was built twice under one id (two residents, two copies of the weights) and
`verify_adapter_among` refused every `--serve-adapters` adapter ("its base ... is served several times"). Identity
must not depend on where the bytes are read from: the copy's `.staged` marker now records the store entry it copies
(`brain_modelstore::stage::staged_entry`), and the scan reads that entry from the copy (`LocalModel::read_from`,
`ScanSettings::copies`), so there is one model, under the store's id, read from node-local disk. A marker without a
source (made before this change) or naming another store is not a copy of this entry and is staged again.

## Evidence (2026-11-22)

- Before: Qwen3-4B on CPU with `--serve-adapters`: plain lists the base and its adapter; staged refuses the adapter.
  The tiny fixture in `brain-cli` `serve_staged` failed the same way.
- After: both list `Qwen/Qwen3-4B` and the adapter (same digest); the old empty-marker copy was restaged (7.0 s).
- Pinned by `brain-cli` `serve_staged::a_staged_checkpoint_is_the_store_entry_it_copies` and `brain-modelstore`
  `stage::tests::concurrent_ranks_stage_one_verified_copy_and_read_from_it`.
