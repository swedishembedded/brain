<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 365. A name-suffix predicate claims a new submodule's same-named leaves

Symptom (caught before it shipped): `qwen35::model::is_quantizable_linear` decides
which tensors a weight tier moves out of the fp32 parameter store with
`name.ends_with("q_proj.weight")` and ten sibling leaves. EdgeRock's anchor memory
adds `blocks.{l}.memory.q_proj.weight`, `k_proj`, `v_proj`, `o_proj` beside the
attention's own, so under any narrow tier those parameters would have been dropped
from the store as "quantizable" while the upload loop (driven by `layer_leaves`,
which does not list them) never uploaded them: a missing-tensor panic at the first
forward, or with a LoRA config an adapter site where none exists. The same shape of
bug waits for the config's `lin()` LoRA helper, which matches on the leaf name.

Cause: leaf names are a vocabulary shared across submodules; a suffix test answers
"does this look like one of the twelve" and not "is this one of the twelve".

Fix: the predicate excludes `.memory.` explicitly, and the memory's parameters are
pushed to `param_list()` by a path that never goes through `lin()`. The durable
answer is to derive such predicates from the table that does the work
(`Qwen35Config::layer_leaves`) instead of a parallel list of suffixes.

Rule: when a new submodule reuses an existing leaf name, grep every `ends_with`/
`contains` on that leaf name before wiring it in, and give the new parameters a
distinct prefix the predicates can exclude.
