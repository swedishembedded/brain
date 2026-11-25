<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 583. A whole listed sampler distribution left a tail of two rounding errors: an empty bucket is no bucket

**Lesson:** the score-centred term decides whether the sampler's unlisted mass is centred by its ratio
`max(1 - sum p, 0) / max(1 - sum q, tiny)`. When the top-K lists the whole distribution, both differences are rounding
errors and the ratio is noise: brain (float64 from float32 log-probabilities) put it inside the band and subtracted
`p_j` from every listed id, while TRL (float32) floored the denominator and decided the other way. The exact estimator
has no tail term then. A listed mass within float32 rounding of one (`1 - sum q <= K * f32::EPSILON`) now leaves no
bucket. The gradient was unchanged in exact arithmetic (`sum_j p_j grad log pi(v_j) = 0` over a whole vocabulary),
the loss value and the coefficients were not.

The comparison that found it reads TRL's own `compute_loss` locals, not a re-derivation: a definition written from
the paper agreed with brain, because both computed in the same precision.

Exactly at a head edge (`p_j = low q_j`) the two still decide by rounding (one ulp of float32 against float64); no
real distribution sits there, and the golden cases stay a hair inside and outside.

## Evidence (2026-11-25)

- `rl` `it::trl_goldens::the_score_centred_term_is_trls`, case `whole_distribution_listed` (q = 0.5, 0.3, 0.2):
  brain tail true, TRL tail false, before the fix; equal after. Unit test
  `score_centring::a_whole_listed_distribution_has_no_tail_whatever_the_rounding`.
