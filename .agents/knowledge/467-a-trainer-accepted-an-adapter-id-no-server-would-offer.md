<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 467. A trainer accepted an adapter id no server would offer

**Lesson:** When one component writes a name and another reads it, both must hold it to the same grammar, and the
writer must refuse a bad name before it spends the work that produces it.

## What happened

`brain/train` checked `adapter_id` only for its characters. A pilot run (2026-11-15) trained three LoRA adapters for
up to an hour each under ids like `lsp-g1-answer-lr1e-4`. `brain serve --serve-adapters` then refused all three with
"is not a model reference": the adapter registry offers an adapter only under `<vendor>/<repo>:<owner>:<name>[:<tag>]`.
The adapters could be served only after their cards were rewritten by hand.

## Fix and evidence

`crates/trainjob/src/request.rs` refuses an `adapter_id` that `ModelRef::parse` does not read as an adapter
reference, naming the grammar, before the run starts. `request::tests::an_adapter_id_is_plain_text` now covers
`lsp-g1` and `Qwen/Qwen3-4B`: it was red before the fix and is green after. The served-training guide states the rule.
