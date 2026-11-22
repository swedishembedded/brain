<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 557. A router the reference keeps in binary32 must not take the default tensor-core tier

**Lesson:** `GemmPrecision::Auto` (the default) rounds a bf16-weight product's activation to bf16 on a tensor-core
device, and runs an f32-weight product as TF32. That is the precision of a bf16 reference run for most products, but
not for the ones the reference keeps in binary32 (`_keep_in_fp32_modules_strict`): a router's top-k turns a rounded
logit into a different expert, and the difference compounds over a deep stack. A model whose reference routes in
fp32 uploads the router as an f32 weight (widened once from the stored bf16) and records its product inside
`model::ops::fp32_kernels`, so it is binary32 at every precision tier. Parity gates pinned to `Fp32Reference` do not
see this; the default path has to be measured against the reference on its own (KL and greedy agreement), beside the
reference's own bf16-run distance.

## Evidence (2026-11-22)

- Inkling 2 Mini (router `mlp.gate.weight` stored bf16, kept f32 by the reference), real checkpoint on a GH200 (CUDA),
  three chat prompts, default arithmetic against the f32 reference golden. Before: visit 47's output 7.5 times the
  bf16 reference run's distance, KL on the full logit rows up to 1.08e-1 nats (bf16 run 6.3e-3), a greedy token the
  f32 reference led by 0.65 logits changed at position 5 of every prompt. After (router binary32): every visit within
  2.1 times the bf16 run's distance, worst KL 2.6e-2 (bf16 run 4.4e-2), one greedy change at a near tie (lead 4.2e-2).
- Making the message gate's f32 products binary32 as well moved the KL by less than the prompt-to-prompt spread
  (2.75e-3 to 2.41e-3, 9.4e-3 to 7.3e-3), so it was not done.
- Gate: `inkling2` `tests/it/real_parity.rs` (the default-arithmetic phase).
