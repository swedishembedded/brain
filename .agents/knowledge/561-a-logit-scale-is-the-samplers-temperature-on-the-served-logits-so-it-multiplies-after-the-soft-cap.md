<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 561. A logit scale is the sampler's temperature on the served logits, so it multiplies after the soft cap

**Lesson:** the logit scale a trainer takes (`TokenObjectives::set_logit_scale`, `1 / T`) corrects for the temperature
the sampler applied, and the sampler divides the logits the model *serves*. For a head with a declared transform
(`HeadTransform`: a soft cap `c tanh(z / c)` and a valid-vocabulary mask) the served logits are `cap(z)`, so the
trained distribution is `softmax(scale * cap(z))`, not `softmax(cap(scale * z))`. `HeadCe` applied the scale first and
the cap after it: at `scale = 1` both agree, which is all its own test and every model without a transform ever ran,
and at any other temperature the trained policy was not the sampled one (the training conformance suite's C1 measured
`-3.3320` against `-3.3197` at scale 0.5 on Inkling 2's capped head). `HeadCe::logits_steps` now caps, then scales;
the backward takes the scale back out of the chunk's scratch logits before the cap's derivative reads them (the scalar
factors commute). A device-held scale (`set_scale_buffer`) beside a transform is refused by name: it could not be taken
back out without a reciprocal kernel, and no model combines the two.

## Evidence (2026-11-22)

- `crates/model/tests/it/head_ce.rs`, `softcap_and_valid_vocab_enter_the_loss_and_its_gradient`: the f64 reference is
  now `scale * softcap(h W^T)` and its gradient; loss, log-probabilities and hidden gradient agree for f32 and bf16
  heads, chunked and windowed.
- `crates/rl/tests/conformance.rs`, `inkling2_conforms`: C1 (log-probabilities equal the log-softmax of the scaled
  served logits at scale 1 and 1/2) passes on the capped head.
