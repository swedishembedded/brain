<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 574. A finite-difference conformance check missed a removed backward term: compare two paths of the model instead

**Lesson:** a new loss term was checked on every registered model by central finite differences of the loss at the
conformance suite's tolerances (`ATOL 4e-3`, `RTOL 8e-2`, best of two random directions). With the qwen35 backward's
wiring of the term removed the check still passed: at a random initialisation the adapters' directional derivatives
were 1e-9 to 1e-3, under the absolute floor, and the f32 finite differences of a loss of a few nats were 0. An
identity between two paths of the same model has no such floor: an extra target `(v, c)` adds exactly what the label
`v` at weight `c` adds, so one backward with extra targets must equal the sum of label-equivalent weighted backwards,
to rounding (`conformance_grad::sparse_targets_match_their_label_equivalents`, 1e-4 of the largest gradient). It failed
with the wiring removed (1.3e-6 off at a largest gradient of 1.1e-3) and passes on all five models with it.

The same holds in a gradient check whose new term is a few percent of the loss: the score-centred estimator's listed
ids carry `-A p_j` against the token's `A r`, so dropping them from the backward left the full-loss check inside its
8% relative tolerance. Checking the new term alone (label weights zero) made the omission fail.

## Evidence (2026-11-24)

- Mutation: `sparse: None` in the qwen35 `HeadBackward`: the finite-difference form passed, the path identity failed.
- Mutation: extra-target coefficients zeroed in the backward of `it::score_centred_gradcheck`: the full-loss checks on
  qwen35 and qwen3 passed, the extra-targets-alone checks failed on both.
