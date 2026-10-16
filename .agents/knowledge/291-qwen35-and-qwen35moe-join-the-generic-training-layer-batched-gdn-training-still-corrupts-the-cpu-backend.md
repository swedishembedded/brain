<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 291. qwen35 and qwen35moe join the generic training layer; batched Gated-DeltaNet training still corrupts the CPU backend

`qwen35::Qwen35` and `qwen35moe::Qwen35` implement `TokenObjectives` and
`LoraSites` and pass the conformance suite (`crates/rl/tests/conformance.rs`:
C1-C4, C6, C7, C9, with the bf16 and NF4 frozen-base tiers for `qwen35`) on the
CUDA backend. How, and what to remember:

- `qwen35` keeps the chunked head: the weighted loss is `RowWeights` on the
  head's recorded backward, the readout is the negated `ce_buf`, the logit scale
  is `HeadCe::set_scale_buffer` plus the scale folded into the gradient factor.
  `crates/qwen35/tests/objectives.rs` checks that a two-chunk head and a
  one-chunk head give the same weighted, scaled loss, log-probabilities and
  gradients, and that the gradient is the derivative of the reported loss.
- `qwen35moe` has no chunked head (it materialises `[n, vocab]` logits), so it
  uses the shared `WeightedCe` row-scaling step and divides the CE gradient by
  `count / scale`. The router and experts sit below the per-token loss, so the
  weights compose with the mixture-of-experts backward untouched; nothing in the
  routing code changed. Only the nine mixer linears carry adapters.
- `Model::new` of both is now the training build (`new_inference` is the old
  forward-only one), `logits_all` accepts any prefix (the model is causal, so
  padding cannot reach the real positions; rollouts ask one prefix after
  another), and `load` rebuilds a model from `save`.
- The NF4 tier's adapter gradients differ from the f32 base's by up to 0.30
  (relative, per tensor): 4-bit weights, so the stated tolerance is 0.4. A
  gradient smaller than 1e-3 of the largest one is not compared: at a random
  initialisation the Gated-DeltaNet `in_proj_qkv` adapters sit behind nearly
  closed gates (|g| about 1e-9) and their relative error is noise.

Open, VERIFIED-BY-RUN: a batch of more than one row through a Gated-DeltaNet
layer's backward aborts with heap corruption on the CPU backend (tiny `qwen35`,
`b = 2`, any `t` tried: 4, 6, 8, 12; with and without LoRA; a single
Gated-DeltaNet layer is enough, a single full-attention layer is fine). The same
build on CUDA gives the finite-difference-correct gradients the suite checks, and
`b = 1` is fine on both. The conformance test for these two models therefore
fails with that reason on the CPU backend instead of aborting. No model test
covered `b > 1` through `gdn_mixer` before this.

Resolved, VERIFIED-BY-RUN: the conv1d input-gradient buffer in
`gdn_mixer_bwd` was sized for one sequence (`t` rows per channel) while the
`conv1d_dx` kernel writes `b * t`. A canary after each CPU dispatch named
`conv1d_dx`. The buffer is now sized for the whole batch;
`crates/qwen35/tests/lora_batch.rs` checks a two-sequence batch against the mean
of its two sequences' gradients, the bf16 CLI fine-tune test is enabled again,
and the qwen35 and qwen35moe conformance subjects pass on the CPU backend.

The suite is slow on these models: about 6 minutes for `qwen35` and `qwen35moe`
together on CUDA with other jobs on the card, almost all of it the
finite-difference check (every trainable tensor, three directions, two scales).
