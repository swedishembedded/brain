<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 579. A snapshot adapter's card named no targets, and no server would serve it: derive them from the model's pairs

**Lesson:** the fit loop writes adapters itself (the evaluation snapshots of `keep_evaluations`, and now the versions
a streaming job publishes) through `lora::save_lora_with_lineage` with the `LoraCfg` its caller passed. The SDK passes
`LoraCfg::new(rank, alpha, Vec::new())`, since it does not build the model: the trainer does. The card then declared
`targets: []` while its tensors corrected seven linears, and the serving bank refused the file ("its card declares
targets [] but its tensors correct [down, gate, up, wk, wo, wq, wv]"). A configuration that names no targets now writes
the leaves the model's own adapter pairs answer to (`Model::adapter_sites`, the leaves the file already records), so the
card names exactly what its tensors correct, for every architecture.

## Evidence (2026-11-24)

- `trainjob` `it::stream::a_stream_job_trains_groups_as_they_arrive_and_serves_each_version_it_publishes` (toy qwen3,
  CPU): the first publication failed with the refusal above before the fix and is listed by `/v1/models` as version 1
  after it.
