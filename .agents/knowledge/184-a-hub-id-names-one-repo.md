<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 184. A hub id names one repo, not an architecture

The SDK's `from_pretrained("<vendor>/<repo>")` resolved through
`loader::resolve_structured`, which scans the **whole** store for the
architecture. The id was parsed only for the download decision and never
used to choose among candidates.

- **With one checkpoint of an architecture in the store, any id loaded it.**
  Asking for `Qwen/Qwen3-0.6B` with only Qwen3-8B downloaded resolved,
  silently, to the 8B.
- **With several, every id was ambiguous.** A store holding the DeepSeek text
  family plus a Qwen3 checkpoint refused
  `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` with seventeen candidates.
- **The candidates were also paired wrongly.** `common_root` infers the store
  root as the records' deepest common ancestor, and specs match a tokenizer
  to weights by the vendor directory under that root. Each candidate carried
  whichever tokenizer the pairing happened to pick, here Qwen3-8B's for every
  DeepSeek directory.

`loader::resolver::resolve_reference` now keeps only the records under
`Store::repo_dir(reference)`. It resolves them with
`brain_modelstore::resolve::resolve_under`, which takes the store root
explicitly, because a set narrowed to one repo has that repo as its common
ancestor and the vendor matching would break.

A reserved vendor (`local/...`) names files dropped into the store with no
repo, so it still resolves over the whole store.
