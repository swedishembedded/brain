<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 270. A logit scale lives on the logits buffer and in the CE gradient's denominator, so it needs no new kernel

An on-policy objective compares the trainer's log-probability of a sampled token
with the sampler's. The sampler draws from `softmax(logits / T)`, so the trainer
must score under `softmax(scale * logits)` with `scale = 1/T`, in the loss, in the
gradient and in the per-token readout.

How `TokenObjectives::set_logit_scale` does it for `qwen3` and `gpt2`, with no new
or edited kernel:

- One in-place `grad_scale_buf` step (`logits *= scale[0]`, scale in a one-float
  device buffer) is appended right after the LM head GEMM. Every consumer of the
  logits then sees scaled values: the CE value pass, the CE statistics, the
  chunked head's backward recompute, `batch_token_logprobs` (the negated `ce_buf`)
  and `logits_all`.
- The CE gradient is `scale * (softmax(scale * z) - onehot) / count`. The extra
  factor is folded into the `1/count` the backward already feeds the gradient
  scaling (qwen3: the `inv_count` device scalar; gpt2: the divisor in the
  `ce_grad_stats` uniform), so the recorded backward steps are unchanged.
- The step exists only after the first scale other than 1, which rebuilds the
  forward and backward tapes once; an unscaled model pays nothing. Later values
  are one scalar write, no rebuild.

Consequences worth remembering:

- `Model::logits_all` returns the scaled logits while a scale is set. Reset the
  scale to 1 before reading raw logits.
- The conformance suite (`model::conformance`) cannot use `gradcheck`'s
  `directional_check`: `gradcheck` depends on `model`, so a library module in
  `model` depending on it would be a cycle. Finite differences in the suite are
  its own few lines over `Model::{read,write}_weight`.
- `rl` objectives cannot live in `model`, so the objective-level conformance
  checks run from `crates/rl/tests/conformance.rs`, which is also where a new
  model is registered.
